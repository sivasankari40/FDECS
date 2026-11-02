# Solari Support Similarity Assistant

When a new support ticket arrives, an agent enters its Ticket ID (or types the subject and description) and sees up to **three similar resolved tickets with their original resolutions** — or a clear **"no strong historical match"** with a suggested team to escalate to. If the closest precedents were resolved differently, the agent is warned to confirm which applies before replying.

- **Design, assumptions and trade-offs:** [SOLUTION_DESIGN.md](SOLUTION_DESIGN.md)
- **Stack:** Python 3.10+, Streamlit UI, CouchDB, scikit-learn TF-IDF + local `all-MiniLM-L6-v2` sentence-transformer (included in `models/`, no download needed)

## How to run it

**Prerequisites:** Python 3.10 or newer, and CouchDB 3.x running locally — either [installed](https://couchdb.apache.org/#download) or started with Docker (`docker compose up -d couchdb`, using [compose.yaml](compose.yaml)).

**Windows (PowerShell)**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env        # then set COUCHDB_USER and COUCHDB_PASSWORD to your CouchDB admin
.\.venv\Scripts\python.exe -m support_assistant.ingest --seed-incoming
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux: avoids GBs of GPU libraries
.venv/bin/python -m pip install -r requirements-dev.txt
cp .env.example .env               # then set COUCHDB_USER and COUCHDB_PASSWORD
.venv/bin/python -m support_assistant.ingest --seed-incoming
.venv/bin/python -m streamlit run app.py
```

Open <http://127.0.0.1:8501>. The ingestion step creates three databases (resolved history, incoming tickets, feedback) and loads the synthetic data from `starter_kit/data/` and `data/`. It is safe to rerun.

## Demo walkthrough

| Enter Ticket ID | What you should see |
|---|---|
| `T-005003` — "Getting charged twice" | `T-001303` (duplicate charge → refund) and `T-002010` (pending authorization → no refund), plus a warning that they were **resolved differently** |
| `T-005002` — blank revenue chart | `T-001201` ranks first |
| `T-003001` — PostgreSQL SSL after renewal | `T-002003` PostgreSQL TLS precedent (~83%) |
| `T-005004` — Snowflake connection | **No strong historical match**, closest score shown, escalate to Data Integration Team |
| `T-005003`, then replace the subject with "Customer billed two times" and the description with "Their bank shows the subscription fee debited twice in one month." | `T-001303` found by meaning — keyword overlap alone is only 9% |

Each result shows the evidence separately (shared terms, semantic similarity, matching tags, exact error codes), a **Why was this suggested?** explanation and the original resolution. Also try:

- **Use resolution** → an editable draft to copy or download (nothing is sent).
- 👍/👎 on a result, then search again → "Earlier agent feedback: helpful 1".
- **Is the right precedent missing?** → enter a Ticket ID; the app explains why it was hidden (e.g. "its similarity score (12%) is below the 30% threshold") and records it.
- Error code field, e.g. `HTTP 500`, and the category filters in the sidebar.

## How it works (short)

- **Ranking:** 0.55 × TF-IDF + 0.45 × semantic similarity, plus a small tag bonus and an exact-error-code boost. Results need a score ≥ 0.30; at most three are shown. Without the model the app falls back to keyword-only matching and says so.
- **Eligibility:** only resolved, approved tickets with notes. In production, the resolution must also have been reviewed within `RESOLUTION_MAX_AGE_DAYS` (180).
- **Error codes:** compared per family — a different code in the same family (`ORA-01017` vs `ORA-00942`) excludes a precedent; tickets without codes are never excluded.
- **Freshness:** CouchDB changes are read incrementally; the index rebuilds in the background; embeddings are cached in `.cache/` so restarts are fast.

Details and reasoning: [SOLUTION_DESIGN.md](SOLUTION_DESIGN.md).

## Tools for support leads

```powershell
python -m support_assistant.feedback --report            # helpful rate by score band, precedents to review, missed precedents by reason, weekly trend
python -m support_assistant.replay                       # re-score real tickets from agent feedback under different thresholds/weights
python -m support_assistant.consistency                  # similar resolved tickets whose resolutions disagree
python -m support_assistant.feedback --export labels.json
```

Triage teams for no-match tickets are configured in [config/triage_routing.json](config/triage_routing.json).

## Tests and evaluation

```powershell
python -m pytest -q                                      # 249 pass; the 3 CouchDB tests are skipped
$env:RUN_COUCHDB_TESTS = '1'; python -m pytest -q        # all 252, using temporary databases on your CouchDB
python -m support_assistant.evaluation --split development
python -m support_assistant.evaluation --split heldout
```

Evaluation uses 32 synthetic scenarios (12 development, 20 held-out). With the production review policy the labels were written for: **Recall@3 1.0, correct abstention 1.0, zero irrelevant results, p95 ≈ 25 ms**. With no age limit (the local default), 40 undated starter tickets and one stale resolution become eligible, and held-out abstention drops to 0.71 — those extra results are tickets the labels assumed were withheld.

## What's not handled yet

- **Helpdesk integration** — incoming tickets are loaded by the ingestion script, not received from Zendesk/Freshdesk.
- **Using feedback to tune** — votes are collected and the replay tool exists, but there is no real feedback yet, so weights and threshold are still tuned on synthetic data.
- **Knowledge-gap / incident trend report** from no-match searches.
- **LLM summaries** of resolutions (deliberately left out; see the design doc).
- **Production readiness:** login is implemented (OIDC with an email allowlist) but untested against a real identity provider; the rate limit is per browser session; privacy masking is regex-based (emails, labelled secrets, tokens), not full PII detection; no backup/restore, load testing or alerting.

## Known limitations

- Similar wording or meaning does not prove the same root cause — the agent must verify every precedent. Scores are ranking signals, not probabilities.
- Threshold (0.30) and weights (55/45) are tuned on 32 synthetic cases written by the same author as the data.
- The index lives in the app process: built for tens of thousands of tickets on one server, not for horizontal scaling.
- One workspace: every agent can see every ticket; there is no per-customer access control.
- Feedback stores ticket IDs only, so searches typed without an incoming Ticket ID can't be replayed; votes are not tied to an agent.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `COUCHDB_URL` | `http://127.0.0.1:5984` | CouchDB server |
| `COUCHDB_USER` / `COUCHDB_PASSWORD` | — | Admin credentials (used by ingestion; the app uses them too in local mode) |
| `COUCHDB_DATABASE` / `COUCHDB_INCOMING_DATABASE` / `COUCHDB_FEEDBACK_DATABASE` | `solari_support_tickets` / `solari_incoming_tickets` / `solari_search_feedback` | Database names |
| `RESOLUTION_MAX_AGE_DAYS` | 0 locally, 180 in production | Review-freshness window (0 = off) |
| `MIN_SIMILARITY` | 0.30 | Recommendation threshold |
| `APP_ENV` | `development` | `production` requires login, a non-admin reader account and HTTPS to a remote CouchDB |

Production-only settings (`COUCHDB_READ_USER`, `AUTH_ALLOWED_EMAILS`, `APP_OIDC_ISSUER`, …) are listed in [.env.example](.env.example); login secrets go in `.streamlit/secrets.toml` (template: [.streamlit/secrets.example.toml](.streamlit/secrets.example.toml)).

## Repository layout

```
app.py                     Streamlit UI
support_assistant/         matching, CouchDB access, ingestion, feedback, consistency, evaluation, replay
config/                    triage routing rules
data/                      24 synthetic history tickets + 32 labelled evaluation scenarios
starter_kit/data/          the provided seed tickets and new-ticket examples
models/all-MiniLM-L6-v2/   sentence-transformer (Apache-2.0, see its LICENSE)
tests/                     unit, UI and CouchDB tests
```

## Troubleshooting

- **"CouchDB denied access"** — check `COUCHDB_USER` / `COUCHDB_PASSWORD` in `.env`, then restart the app.
- **"The ticket database does not exist"** — run the ingestion command.
- **"Semantic model unavailable"** — the `models/` folder is missing or incomplete; the app still works with keyword matching. Restore it with `python -m support_assistant.semantic` (downloads ~90 MB).

## Credits

Seed data from the case-study starter kit. Sentence-transformer [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache License 2.0 — licence text in `models/all-MiniLM-L6-v2/LICENSE`.
