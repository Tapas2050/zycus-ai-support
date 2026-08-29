import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.triage_agent import TriageAgent, TriageResult
from app.models import TicketInput


agent = TriageAgent()

ticket = TicketInput(
    subject="AnalyticsHub dashboard takes more than 100 seconds to load",
    body=(
        "Our AnalyticsHub dashboard is taking more than 100 seconds to load. "
        "This is affecting our users and making the dashboard very difficult "
        "to use. Please help us investigate."
    ),
)

# ---------------------------------------------------------
# STEP 1: SEE WHAT THE RETRIEVER FINDS
# ---------------------------------------------------------

query = f"Subject: {ticket.subject}\n\nBody:\n{ticket.body}"

retrieved = agent.kb.retrieve(
    query,
    top_k=5,
)

print("\n===== RETRIEVED KB =====")

for rank, (chunk, score) in enumerate(retrieved, start=1):
    print(f"\n--- RESULT {rank} ---")
    print(f"Score: {score:.3f}")
    print(f"File: {chunk.source_file}")
    print(f"Heading: {chunk.heading}")
    print(f"Text:\n{chunk.text}")

print("\n===== TRIAGE JSON SCHEMA =====")
print(TriageResult.model_json_schema())

# ---------------------------------------------------------
# STEP 2: RUN THE ACTUAL AGENT
# ---------------------------------------------------------

result = agent.triage(ticket)

print("\n===== TRIAGE RESULT =====")
print(result.model_dump_json(indent=2))