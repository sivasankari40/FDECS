# Starter Kit — Support Ticket Similarity Assistant

This folder contains the raw materials for the case study — none of it is meant to hand you the solution, it's just here so you're not spending your limited time on boilerplate.

## Contents

- `data/tickets.json` — ~40 synthetic historical support tickets (subject, description, resolution notes, tags, category, created date).
- `data/new_tickets_example.json` — a handful of new incoming tickets you can use to test your solution end-to-end, including one deliberately unusual case with no strong historical match.
- `ticket_utils.py` (removed) — the optional naive keyword-overlap utility; the solution replaced it with `support_assistant/matching.py`.

## A few notes

- The dataset is intentionally synthetic and modest in size — you shouldn't need to worry about real-world data cleaning; focus your time on the actual design and matching logic.
- `data/new_tickets_example.json` includes one ticket about a Snowflake integration failure that doesn't closely resemble anything in the historical set on purpose — how your solution handles a "no good match" case is worth thinking about.
