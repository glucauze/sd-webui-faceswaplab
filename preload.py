from argparse import ArgumentParser


def preload(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--faceswaplab_loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    print("FACESWAPLAB================================================================")
    print("BREAKING CHANGE: enforce face checkpoint format from pkl to safetensors\n")
    print("Using pkl files to store faces is dangerous from a security point of view.")
    print("For the same reason that models are now stored in safetensors,")
    print("We are switching to safetensors for the storage format.")
    print(
        "A script with instructions for converting existing pkl files can be found here:"
    )
    print("https://gist.github.com/glucauze/4a3c458541f2278ad801f6625e5b9d3d")
    print("==========================================================================")
