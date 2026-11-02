import argparse
import sys
import json
from dataclasses import replace
from pathlib import Path

from support_assistant.config import ROOT, Settings
from support_assistant.feedback import prepare_feedback_database
from support_assistant.repository import CouchRepository, RepositoryError, load_history
from support_assistant.models import Ticket


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and import historical support tickets into CouchDB.")
    parser.add_argument("--file", type=Path, nargs="+", default=[ROOT / "starter_kit" / "data" / "tickets.json", ROOT / "data" / "production_tickets.json"])
    parser.add_argument("--provision-reader", action="store_true", help="Explicitly create/configure a non-admin read account and database write guard; requires CouchDB server-admin credentials.")
    parser.add_argument("--seed-incoming", action="store_true", help="Bootstrap the synthetic incoming examples into CouchDB; never overwrite existing IDs.")
    parser.add_argument("--initialize-only", action="store_true", help="Create empty ticket and feedback databases without reading fixture data.")
    args = parser.parse_args()
    try:
        tickets = [] if args.initialize_only else [ticket for path in args.file for ticket in load_history(path)]
        if len({ticket.ticket_id for ticket in tickets}) != len(tickets):
            raise ValueError("Duplicate ticket IDs across input files.")
        settings = Settings.from_env()
        incoming_settings = replace(settings, database=settings.incoming_database, incoming_database=settings.database)
        with CouchRepository(settings) as repository, CouchRepository(incoming_settings) as incoming_repository:
            if args.initialize_only:
                repository.initialize()
            incoming_repository.initialize()
            if args.provision_reader:
                repository.provision_reader()
                incoming_repository.provision_reader()
            resolved = [ticket for ticket in tickets if ticket.status == "resolved"]
            report = repository.import_tickets(resolved)
            repository.enforce_store_kind("resolved")
            incoming_repository.enforce_store_kind("incoming")
            prepare_feedback_database(settings)
            if args.seed_incoming:
                with (ROOT / "starter_kit/data/new_tickets_example.json").open(encoding="utf-8") as source:
                    examples = json.load(source)["new_tickets"]
                with (ROOT / "data/evaluation_cases.json").open(encoding="utf-8") as source:
                    examples += [{"ticket_id": case["case_id"], "subject": case["subject"], "description": case["description"], "error_codes": case.get("error_codes")}
                                 for case in json.load(source)["cases"]]
                existing_ids = {ticket.ticket_id for ticket in incoming_repository.list_tickets()}
                incoming = [Ticket.incoming(example["ticket_id"], example["subject"], example["description"],
                                            error_codes=tuple(example["error_codes"]) if example.get("error_codes") is not None else None)
                            for example in examples]
                fresh = [ticket for ticket in incoming if ticket.ticket_id not in existing_ids]
                incoming_report = incoming_repository.import_tickets(fresh)
                print(f"Incoming demo bootstrap: {incoming_report.created} created; existing tickets left unchanged.")
        print(f"Historical input {len(resolved)} resolved tickets: {report.created} created, {report.updated} updated, {report.unchanged} unchanged, {report.skipped_deleted} intentionally deleted IDs skipped.")
        return 0
    except (ValueError, RepositoryError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())