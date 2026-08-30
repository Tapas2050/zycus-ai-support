import sys

from app.health_agent import HealthAgent


# ---------------------------------------------------------
# STEP 1: CREATE THE HEALTH AGENT
# ---------------------------------------------------------

agent = HealthAgent()


# ---------------------------------------------------------
# STEP 2: READ ACCOUNT ID FROM COMMAND LINE
# ---------------------------------------------------------

if len(sys.argv) != 2:
    print(
        "Usage: python health_check.py <account_id>"
    )
    print(
        "Example: python health_check.py ACC-3336"
    )
    sys.exit(1)

account_id = sys.argv[1]


# ---------------------------------------------------------
# STEP 3: VERIFY THAT THE ACCOUNT EXISTS
# ---------------------------------------------------------

account = agent.repo.get_account(account_id)

if account is None:
    print(f"\nUnknown account_id: {account_id}")
    print("\nAvailable account IDs:")

    for item in agent.repo.accounts:
        print(f"- {item.account_id}")

    sys.exit(1)


print("\n===== SELECTED ACCOUNT =====")
print(f"Account ID: {account.account_id}")
print(f"Company: {account.company}")
print(f"Health: {account.health_status}")
print(f"ARR: ${account.arr_usd}")


# ---------------------------------------------------------
# STEP 4: INSPECT TICKET HISTORY
# ---------------------------------------------------------

tickets, join_strategy = agent.repo.get_tickets_for_account(
    account_id,
    days=90,
)

print("\n===== TICKET HISTORY =====")
print(f"Join strategy: {join_strategy}")
print(f"Tickets found: {len(tickets)}")

for ticket in tickets:
    print("\n--- TICKET ---")
    print(f"Ticket ID: {ticket.ticket_id}")
    print(f"Created: {ticket.created_at}")
    print(f"Company: {ticket.company}")
    print(f"Subject: {ticket.subject}")
    print(f"Status: {ticket.status}")
    print(f"Urgency: {ticket.urgency}")
    print(f"Category: {ticket.category}")
    print(f"CSAT: {ticket.satisfaction_score}")
    print(f"Body: {ticket.body}")


# ---------------------------------------------------------
# STEP 5: RUN THE HEALTH AGENT
# ---------------------------------------------------------

print("\n===== RUNNING HEALTH AGENT =====")

result = agent.summarise(account_id)


# ---------------------------------------------------------
# STEP 6: DISPLAY FINAL RESULT
# ---------------------------------------------------------

print("\n===== ACCOUNT HEALTH RESULT =====")
print(result.model_dump_json(indent=2))