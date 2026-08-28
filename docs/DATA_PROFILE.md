# Supplied Dataset Profile

## Corpus

- 500 synthetic support tickets
- 50 synthetic customer accounts
- 9 Markdown knowledge-base documents
- 106 heading-level KB chunks (derived from the supplied Markdown headings)

## Ticket distribution

- Products: SecureVault 123, WorkflowEngine 103, AnalyticsHub 101, DataBridge Pro 89, CloudSync 84
- Categories: Data Loss 73, Feature Request 71, Performance 68, How-To 66, Onboarding 63, Bug 59, Billing 50, Integration 50
- Urgency labels: P3 217, P4 159, P2 110, P1 14
- Status: Resolved 164, Closed 132, In Progress 91, Open 68, Pending Customer 45
- Channels: phone 133, portal 130, email 121, chat 116

## Important data-quality findings

### 1. Ticket/account IDs are not a reliable join in this corpus

Only 4 of the 500 ticket `account_id` values match an ID in `accounts.json`. However, the ticket `company` values map to the 50 account companies.

The implementation therefore:
1. tries `ticket.account_id == account.account_id`;
2. validates that matched tickets have the same company as the requested account;
3. if the ID join is absent or inconsistent, falls back to `ticket.company == account.company`;
4. records the join strategy so the behavior is observable.

We do not mutate or rewrite the supplied data.

### 2. The ticket corpus is a historical 90-day window

Ticket timestamps span approximately 90 days, from 2026-02-20 to 2026-05-22. Therefore the default account-health `as_of` is the latest ticket timestamp, rather than the machine's current date. This keeps the supplied offline dataset meaningful.

`DATA_AS_OF` can override this behavior.

### 3. Ticket label fields should not be treated as perfect semantic truth

The synthetic records contain obvious cases where the structured category/urgency metadata conflicts with the natural-language ticket. For example, some tickets explicitly say “P1” while their stored urgency is not P1, and some subjects clearly describe billing/onboarding/integration while the stored category differs.

For the AI triage task, the model should classify the incoming ticket from its content plus retrieved knowledge, not simply copy historical label fields.

For evaluation, expected outputs should therefore use explicit acceptance criteria grounded in the ticket text and KB rather than blindly asserting that every historical label is correct.

## Risk-signal sources

Account-level risk signals are available in:
- `health_status`
- `usage_trend`
- `open_tickets`
- `p1_tickets_last_30d`
- `escalation_notes`
- `nps_score`
- `last_login_days_ago`
- renewal timing

Ticket-level evidence can include:
- explicit urgency/business-impact language
- production impact
- unresolved/open status
- repeated failures
- low CSAT
- explicit escalation language

The health summariser must quote the ticket directly when flagging a ticket as a churn/escalation signal, as required by the assignment.
