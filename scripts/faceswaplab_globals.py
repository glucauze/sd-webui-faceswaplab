import os

MODELS_DIR = os.path.abspath(os.path.join("models", "faceswaplab"))
ANALYZER_DIR = os.path.abspath(os.path.join(MODELS_DIR, "analysers"))
FACE_PARSER_DIR = os.path.abspath(os.path.join(MODELS_DIR, "parser"))

VERSION_FLAG: str = "v1.1.0"
EXTENSION_PATH = os.path.join("extensions", "sd-webui-faceswaplab")

# The NSFW score threshold. If any part of the image has a score greater than this threshold, the image will be considered NSFW.
NSFW_SCORE_THRESHOLD: float = 0.7
