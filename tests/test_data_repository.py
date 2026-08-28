import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from app.data_repository import DataRepository


def test_dataset_shape():
    repo = DataRepository("data")
    assert len(repo.tickets) == 500
    assert len(repo.accounts) == 50


def test_dataset_as_of_is_latest_ticket():
    repo = DataRepository("data")
    assert repo.dataset_as_of == max(t.created_at for t in repo.tickets)


def test_inconsistent_account_id_uses_company_fallback():
    repo = DataRepository("data")
    tickets, strategy = repo.get_tickets_for_account("ACC-3336")
    assert strategy == "company_fallback"
    assert tickets
    assert all(t.company == "Omni Consumer Products" for t in tickets)


def test_unknown_account_is_graceful():
    repo = DataRepository("data")
    tickets, strategy = repo.get_tickets_for_account("ACC-DOES-NOT-EXIST")
    assert tickets == []
    assert strategy == "account_not_found"
