from app.data_repository import DataRepository
from app.kb_retriever import KBRetriever


def main() -> None:
    repo = DataRepository()
    kb = KBRetriever()

    print(f"Tickets: {len(repo.tickets)}")
    print(f"Accounts: {len(repo.accounts)}")
    print(f"Dataset as-of: {repo.dataset_as_of.isoformat()}")
    print(f"KB chunks: {len(kb.chunks)}")

    matches, strategy = repo.get_tickets_for_account("ACC-3336")
    print(f"ACC-3336 ticket join: {strategy}; tickets={len(matches)}")

    results = kb.retrieve(
        "DataBridge Pro connector authentication failure ERR_CONNECTION_TIMEOUT",
        product="DataBridge Pro",
    )
    for chunk, score in results:
        print(f"{score:.3f} {chunk.source_file} :: {chunk.heading}")


if __name__ == "__main__":
    main()