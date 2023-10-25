"""
commons.py

Common functionality and variable constants used across the entire SW package.
"""

from datetime import datetime
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
ATTACK_DATA_DIR = os.path.join(SCRIPT_DIR, "attack_data")

# The data root dir on the cluster
EXP_ROOT_DATA_DIR = "/home/zahalja1/data"
EXP_POISON_RATES = [0.1, 0.2]


def timestamp() -> str:
    """
    A timestamp for the SW package's console output.
    """
    return f"[{datetime.now().strftime('%d %b %Y, %H:%M:%S')}]"


def t_readable(seconds):
    """
    Converts time in seconds to a more easily-readable format of days, hours,
    minutes, and seconds.
    """
    days = int(seconds // (24 * 60 * 60))
    seconds %= 24 * 60 * 60

    hours = int(seconds // (60 * 60))
    seconds %= 60 * 60

    minutes = int(seconds // 60)
    seconds = round(seconds % 60, 2)

    if days > 0:
        tstamp = f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"
    elif hours > 0:
        tstamp = f"{hours} hours, {minutes} minutes, {seconds} seconds"
    elif minutes > 0:
        tstamp = f"{minutes} minutes, {seconds} seconds"
    else:
        tstamp = f"{seconds} seconds"

    return tstamp
