import logging
import copy
import sys
from typing import Any
from modules import shared
from PIL import Image
from logging import LogRecord


class ColoredFormatter(logging.Formatter):
    """
    A custom logging formatter that outputs logs with level names colored.

    Class Attributes:
        COLORS (dict): A dictionary mapping logging level names to their corresponding color codes.

    Inherits From:
        logging.Formatter
    """

    COLORS: dict[str, str] = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record: LogRecord) -> str:
        """
        Format the specified record as text.

        The record's attribute dictionary is used as the operand to a string
        formatting operation which yields the returned string. Before formatting
        the dictionary, a check is made to see if the format uses the levelname
        of the record. If it does, a colorized version is created and used.

        Args:
            record (LogRecord): The log record to be formatted.

        Returns:
            str: The formatted string which includes the colorized levelname.
        """
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Create a new logger
logger = logging.getLogger("FaceSwapLab")
logger.propagate = False

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

# Configure logger
loglevel_string = getattr(shared.cmd_opts, "faceswaplab_loglevel", "INFO")
loglevel = getattr(logging, loglevel_string.upper(), "INFO")
logger.setLevel(loglevel)

import tempfile

if logger.getEffectiveLevel() <= logging.DEBUG:
    DEBUG_DIR = tempfile.mkdtemp()


def save_img_debug(img: Image.Image, message: str, *opts: Any) -> None:
    """
    Saves an image to a temporary file if the logger's effective level is set to DEBUG or lower.
    After saving, it logs a debug message along with the file URI of the image.

    Parameters
    ----------
    img : Image.Image
        The image to be saved.
    message : str
        The message to be logged.
    *opts : Any
        Additional arguments to be passed to the logger's debug method.

    Returns
    -------
    None
    """
    if logger.getEffectiveLevel() <= logging.DEBUG:
        with tempfile.NamedTemporaryFile(
            dir=DEBUG_DIR, delete=False, suffix=".png"
        ) as temp_file:
            img_path = temp_file.name
            img.save(img_path)

        message_with_link = f"{message}\nImage: file://{img_path}"
        logger.debug(message_with_link, *opts)
