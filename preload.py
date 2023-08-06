from argparse import ArgumentParser


def preload(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--faceswaplab_loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--faceswaplab_gpu",
        action="store_true",
        help="Enable GPU if set, disable if not set",
    )
