import os
from modules import scripts
from modules.shared import opts

# Defining the absolute path for the 'faceswaplab' directory inside 'models' directory
MODELS_DIR = os.path.abspath(os.path.join("models", "faceswaplab"))
# Defining the absolute path for the 'analysers' directory inside 'MODELS_DIR'
ANALYZER_DIR = os.path.abspath(os.path.join(MODELS_DIR, "analysers"))
# Defining the absolute path for the 'parser' directory inside 'MODELS_DIR'
FACE_PARSER_DIR = os.path.abspath(os.path.join(MODELS_DIR, "parser"))
# Defining the absolute path for the 'faces' directory inside 'MODELS_DIR'
FACES_DIR = os.path.abspath(os.path.join(MODELS_DIR, "faces"))

# Constructing the path for 'references' directory inside the 'extensions' and 'sd-webui-faceswaplab' directories, based on the base directory of scripts
REFERENCE_PATH = os.path.join(
    scripts.basedir(), "extensions", "sd-webui-faceswaplab", "references"
)

# Defining the version flag for the application
VERSION_FLAG: str = "v1.2.2"
# Defining the path for 'sd-webui-faceswaplab' inside the 'extensions' directory
EXTENSION_PATH = os.path.join("extensions", "sd-webui-faceswaplab")

# Defining the NSFW score threshold. Any image part with a score above this value will be treated as NSFW (Not Safe For Work)
NSFW_SCORE_THRESHOLD: float = opts.data.get("faceswaplab_nsfw_threshold", 0.7)  # type: ignore
# Defining the expected SHA1 hash value for 'INSWAPPER'
EXPECTED_INSWAPPER_SHA1 = "17a64851eaefd55ea597ee41e5c18409754244c5"
