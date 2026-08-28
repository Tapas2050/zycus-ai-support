import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent / "src"),
)
import json

import gradio as gr
from fastapi.testclient import TestClient

from app.api import app as fastapi_app
from app.health_agent import AccountHealthResult
from app.models import TicketInput
from app.triage_agent import TriageResult

# All UI calls go through the real FastAPI app in-process via TestClient,
# instead of importing and calling the agents directly. This exercises the
# exact same routing, request validation, and HTTPException handling that a
# deployed uvicorn process would use, so the UI and the API layer can never
# silently drift apart. See docs/ARCHITECTURE.md for more detail.
client = TestClient(fastapi_app)

# -----------------------------------------------------------------------
# Static reference data / lookups
# -----------------------------------------------------------------------

URGENCY_BADGE = {
    "p1": "🔴",
    "p2": "🟠",
    "p3": "🟡",
    "p4": "🟢",
}

SEVERITY_BADGE = {
    "high": "🔴",
    "medium": "🟠",
    "low": "🟡",
}

EXAMPLE_TICKETS = [
    [
        "SSO login failing for all users since this morning",
        "Since roughly 9am today none of our users can log in via SSO. "
        "They get redirected back to the login page with no error message. "
        "This is blocking our entire team, please treat as urgent.",
    ],
    [
        "Question about upgrading our plan tier",
        "We're currently on the Professional plan and want to understand "
        "what changes with Enterprise, and whether there's a self-serve way "
        "to upgrade or if we need to talk to sales.",
    ],
    [
        "DataBridge Pro sync intermittently dropping records",
        "Over the last week our nightly sync via DataBridge Pro has been "
        "dropping a small percentage of records without any error in the logs. "
        "It's inconsistent - happens maybe 1 in 10 runs.",
    ],
]


def _account_choices() -> list[tuple[str, str]]:
    """Build (label, value) choices for the account dropdown via the API."""
    try:
        resp = client.get("/accounts")
        resp.raise_for_status()
        return sorted(
            (f"{a['account_id']} — {a['company']}", a["account_id"])
            for a in resp.json()
        )
    except Exception:
        return []


# -----------------------------------------------------------------------
# Formatting helpers — turn a pydantic result into readable Markdown
# -----------------------------------------------------------------------

def _format_triage_markdown(result) -> str:
    urgency = result.urgency.lower()
    badge = URGENCY_BADGE.get(urgency, "⚪")

    lines = [
        f"### {badge} {result.urgency.upper()} · {result.category}",
        "",
        f"**Product:** {result.product or 'Unspecified'}  ",
        f"**Product area:** {result.product_area}  ",
        f"**Route to:** {result.recommended_responder_team}  ",
        f"**Known issue:** {'Yes' if result.known_issue else 'No'}",
        "",
        "**Rationale**",
        result.rationale,
        "",
        "**Suggested first response**",
        f"> {result.first_response}",
    ]

    if result.knowledge_base_matches:
        lines.append("")
        lines.append("**Knowledge base matches**")
        for m in result.knowledge_base_matches:
            heading = f" — {m.heading}" if m.heading else ""
            lines.append(f"- `{m.source_file}`{heading}: {m.relevance_reason}")

    lines.append("")
    lines.append(f"<sub>prompt version: {result.prompt_version}</sub>")
    return "\n".join(lines)


def _format_health_markdown(result) -> str:
    lines = [
        f"### Account {result.account_id}",
        "",
        result.executive_summary,
    ]

    if result.account_level_risks:
        lines.append("")
        lines.append("**Account-level risks**")
        for r in result.account_level_risks:
            badge = SEVERITY_BADGE.get(r.severity.lower(), "⚪")
            lines.append(f"- {badge} **{r.signal}** ({r.severity}) — {r.reasoning}")

    if result.ticket_risks:
        lines.append("")
        lines.append("**Ticket-level risks**")
        for r in result.ticket_risks:
            badge = SEVERITY_BADGE.get(r.severity.lower(), "⚪")
            lines.append(
                f"- {badge} `{r.ticket_id}` **{r.risk_type}** ({r.severity}) — {r.reasoning}"
            )
            lines.append(f'  > "{r.evidence_quote}"')

    if result.talking_points:
        lines.append("")
        lines.append("**Talking points for next call**")
        for t in result.talking_points:
            lines.append(f"- {t}")

    lines.append("")
    lines.append(f"*Ticket join strategy: {result.ticket_join_strategy}*")
    lines.append("")
    lines.append(f"<sub>prompt version: {result.prompt_version}</sub>")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------

def triage_ticket(subject: str, body: str):
    if not subject.strip():
        raise gr.Error("Please enter a ticket subject.")
    if not body.strip():
        raise gr.Error("Please enter the ticket body.")

    try:
        ticket = TicketInput(subject=subject, body=body)
        resp = client.post("/triage", json=ticket.model_dump())
        if resp.status_code != 200:
            raise RuntimeError(resp.json().get("detail", resp.text))
        result = TriageResult.model_validate(resp.json())
    except Exception as exc:
        raise gr.Error(f"Triage failed: {exc}") from exc

    markdown = _format_triage_markdown(result)
    raw_json = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
    return markdown, raw_json


def clear_triage():
    return "", "", "", ""


def _format_ticket_history_markdown(account_id: str) -> str:
    resp = client.get(f"/accounts/{account_id.strip()}/tickets")
    if resp.status_code != 200:
        return f"_Could not load ticket history: {resp.json().get('detail', resp.text)}_"

    payload = resp.json()
    tickets = payload["tickets"]
    join_strategy = payload["join_strategy"]

    if not tickets:
        return "_No tickets found for this account in the lookback window._"

    lines = [f"**{len(tickets)} ticket(s)** · join: `{join_strategy}`", ""]
    for t in tickets:
        badge = URGENCY_BADGE.get(t["urgency"].lower(), "⚪")
        date = t["created_at"][:10]
        lines.append(
            f"- {badge} `{t['ticket_id']}` **{t['subject']}** — {t['status']} "
            f"({t['category']}, {date})"
        )
    return "\n".join(lines)


def get_account_health(account_id: str):
    if not account_id or not account_id.strip():
        raise gr.Error("Please choose an account.")

    try:
        resp = client.get(f"/accounts/{account_id.strip()}/health")
        if resp.status_code == 404:
            raise ValueError(resp.json().get("detail", "not found"))
        if resp.status_code != 200:
            raise RuntimeError(resp.json().get("detail", resp.text))
        result = AccountHealthResult.model_validate(resp.json())
    except ValueError as exc:
        raise gr.Error(f"Account not found: {exc}") from exc
    except Exception as exc:
        raise gr.Error(f"Account health analysis failed: {exc}") from exc

    markdown = _format_health_markdown(result)
    raw_json = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
    history = _format_ticket_history_markdown(account_id)
    return markdown, raw_json, history


def clear_health():
    return None, "", "", ""


# -----------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------

CUSTOM_CSS = """
:root {
    --zy-accent: #2563EB;
    --zy-radius: 6px;
}
.gradio-container {max-width: 1080px !important; margin: auto; font-family: 'Inter', -apple-system, sans-serif;}
#app_header {text-align: left; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-color-primary);}
#app_header h1 {margin-bottom: 0.15rem; font-weight: 700; letter-spacing: -0.02em;}
#app_header p {color: var(--body-text-color-subdued); margin: 0;}
.result_panel {border: 1px solid var(--border-color-primary); border-radius: var(--zy-radius); padding: 1.25rem; background: var(--background-fill-secondary);}
button.primary {background: var(--zy-accent) !important; border-radius: var(--zy-radius) !important; transition: opacity 200ms ease;}
button.primary:hover {opacity: 0.88;}
.gr-button {border-radius: var(--zy-radius) !important; transition: all 200ms ease;}
input, textarea, select {border-radius: var(--zy-radius) !important;}
.tab-nav button {font-weight: 600;}
footer {visibility: hidden}
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    radius_size="sm",
)

with gr.Blocks(
    title="Zycus AI Support Assistant",
) as demo:

    gr.Markdown(
        """
        # Zycus AI Support Assistant
        <p>AI-assisted support ticket triage and customer account health analysis.</p>
        """,
        elem_id="app_header",
    )

    with gr.Tabs():

        # ------------------------------------------------------------
        with gr.Tab("🎫 Ticket Triage"):
            gr.Markdown(
                "Paste in a support ticket and get product/category/urgency "
                "classification, routing, and a suggested first response."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    subject = gr.Textbox(
                        label="Ticket Subject",
                        placeholder="Enter the support ticket subject...",
                    )
                    body = gr.Textbox(
                        label="Ticket Body",
                        placeholder="Enter the support ticket description...",
                        lines=10,
                    )
                    with gr.Row():
                        triage_button = gr.Button("Analyze Ticket", variant="primary")
                        triage_clear = gr.Button("Clear")

                    gr.Examples(
                        examples=EXAMPLE_TICKETS,
                        inputs=[subject, body],
                        label="Try an example",
                    )

                with gr.Column(scale=1):
                    triage_markdown = gr.Markdown(
                        label="Triage Result",
                        elem_classes=["result_panel"],
                    )
                    with gr.Accordion("Raw JSON", open=False):
                        triage_json = gr.Code(language="json", show_label=False)

            triage_button.click(
                fn=triage_ticket,
                inputs=[subject, body],
                outputs=[triage_markdown, triage_json],
                api_name="triage_ticket",
            )
            triage_clear.click(
                fn=clear_triage,
                inputs=None,
                outputs=[subject, body, triage_markdown, triage_json],
            )

        # ------------------------------------------------------------
        with gr.Tab("📊 Account Health"):
            gr.Markdown(
                "Pick a customer account to get a TAM-ready health summary: "
                "risk signals, at-risk tickets, and talking points."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    account_id = gr.Dropdown(
                        label="Account",
                        choices=_account_choices(),
                        filterable=True,
                        info="Search by account ID or company name",
                    )
                    with gr.Row():
                        health_button = gr.Button("Analyze Account", variant="primary")
                        health_clear = gr.Button("Clear")

                with gr.Column(scale=1):
                    health_markdown = gr.Markdown(
                        label="Account Health Result",
                        elem_classes=["result_panel"],
                    )
                    with gr.Accordion("Recent Tickets", open=True):
                        ticket_history_markdown = gr.Markdown()
                    with gr.Accordion("Raw JSON", open=False):
                        health_json = gr.Code(language="json", show_label=False)

            health_button.click(
                fn=get_account_health,
                inputs=account_id,
                outputs=[health_markdown, health_json, ticket_history_markdown],
                api_name="account_health",
            )
            health_clear.click(
                fn=clear_health,
                inputs=None,
                outputs=[account_id, health_markdown, health_json, ticket_history_markdown],
            )

    gr.Markdown(
        "<center><sub>Zycus AI Support Assistant — internal tool, results are "
        "LLM-generated and should be reviewed before acting on them.</sub></center>"
    )


if __name__ == "__main__":
    # For a shared/public deploy, gate access with basic auth. Set
    # UI_USER / UI_PASSWORD in .env (never hardcode credentials here),
    # then uncomment the `auth=` line below.
    #
    # import os
    # auth = (os.getenv("UI_USER"), os.getenv("UI_PASSWORD"))
    demo.queue().launch(theme=THEME, css=CUSTOM_CSS)