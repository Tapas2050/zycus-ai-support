import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.models import Account, Ticket


class DataRepository:
    """Read-only access to the supplied synthetic datasets.

    The starter data contains an intentional data-quality problem:
    ticket.account_id is frequently inconsistent with accounts.json.
    The ticket company field, however, maps to the account company.
    We therefore use account_id as the primary join and company as a
    validated fallback only when the ID join yields no ticket history.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._tickets = self._load_tickets()
        self._accounts = self._load_accounts()

        self.accounts_by_id = {a.account_id: a for a in self._accounts}
        self.accounts_by_company = {a.company: a for a in self._accounts}

        self.dataset_as_of = max(t.created_at for t in self._tickets)

    def _load_tickets(self) -> list[Ticket]:
        with (self.data_dir / "tickets.json").open(encoding="utf-8") as f:
            return [Ticket.model_validate(x) for x in json.load(f)]

    def _load_accounts(self) -> list[Account]:
        with (self.data_dir / "accounts.json").open(encoding="utf-8") as f:
            return [Account.model_validate(x) for x in json.load(f)]

    @property
    def tickets(self) -> list[Ticket]:
        return list(self._tickets)

    @property
    def accounts(self) -> list[Account]:
        return list(self._accounts)

    def get_account(self, account_id: str) -> Account | None:
        return self.accounts_by_id.get(account_id)

    def get_tickets_for_account(
        self,
        account_id: str,
        days: int = 90,
        as_of: datetime | None = None,
    ) -> tuple[list[Ticket], str]:
        """Return recent tickets and explain which join strategy was used.

        The synthetic ticket corpus spans roughly one 90-day window. For an
        offline dataset, using max(created_at) as the default as-of point
        avoids incorrectly comparing the historical corpus with today's clock.
        """
        account = self.get_account(account_id)
        if account is None:
            return [], "account_not_found"

        reference = as_of or self.dataset_as_of
        cutoff = reference - timedelta(days=days)

        direct = [
            t for t in self._tickets
            if t.account_id == account_id
            and cutoff <= t.created_at <= reference
        ]

        # The supplied corpus can contain an account_id collision where the
        # ID points at a different customer's ticket. Accept the direct join
        # only when the ticket company also agrees with the account.
        consistent_direct = [t for t in direct if t.company == account.company]
        if consistent_direct:
            return sorted(consistent_direct, key=lambda t: t.created_at, reverse=True), "account_id"

        # Data-quality fallback: the supplied corpus consistently carries
        # the customer company on the ticket. Use it when the ID join is
        # absent or inconsistent with the account's company.
        by_company = [
            t for t in self._tickets
            if t.company == account.company
            and cutoff <= t.created_at <= reference
        ]
        return sorted(by_company, key=lambda t: t.created_at, reverse=True), "company_fallback"
