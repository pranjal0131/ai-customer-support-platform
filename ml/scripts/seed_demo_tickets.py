"""Seed the configured database with deterministic demo tickets."""

from backend.app.database import SessionLocal, init_db
from backend.app.ml_service import ModelService
from backend.app.seed import seed_if_empty


def main() -> None:
    init_db()
    with SessionLocal() as session:
        count = seed_if_empty(session, ModelService())
    print(f"Seeded {count} tickets (0 means data already existed).")


if __name__ == "__main__":
    main()
