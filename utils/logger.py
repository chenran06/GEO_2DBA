"""Logging helpers replacing the print-tee hack of the legacy scripts."""
import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name, log_dir, filename):
    """Create a logger writing to both stdout and ``log_dir/filename``.

    Args:
        name: logger name (use the script name to keep loggers separate).
        log_dir: directory the log file is written to (created if missing).
        filename: log file name; typically a timestamp-prefixed stem.

    Returns:
        logging.Logger with one console handler and one file handler.
        Repeated calls with the same name return the existing logger
        instead of duplicating handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:  # already configured (e.g. module re-import)
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def format_metrics(metrics, indent="  "):
    """Render a flat metrics dict as aligned ``key: value`` lines.

    Args:
        metrics: dict of metric name -> number.
        indent: leading whitespace of every line.
    Returns:
        Multi-line string, one metric per line.
    """
    width = max((len(k) for k in metrics), default=0)
    lines = [f"{indent}{k:<{width}} : {v:.4f}" for k, v in metrics.items()]
    return "\n".join(lines)
