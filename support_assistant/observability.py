import json
import logging
from datetime import datetime, timezone


class EventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"time": datetime.now(timezone.utc).isoformat(), "level": record.levelname,
                           "event": record.getMessage(), **getattr(record, "metrics", {})}, allow_nan=False)


logger = logging.getLogger("solari.events")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(EventFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def event(name: str, **metrics: int | float | bool) -> None:
    logger.info(name, extra={"metrics": metrics})