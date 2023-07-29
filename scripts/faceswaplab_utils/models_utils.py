import glob
import os
from typing import List
import modules.scripts as scripts
from modules import scripts
from scripts.faceswaplab_globals import EXTENSION_PATH
from modules.shared import opts
from scripts.faceswaplab_utils.faceswaplab_logging import logger


def get_models() -> List[str]:
    """
    Retrieve a list of swap model files.

    This function searches for model files in the specified directories and returns a list of file paths.
    The supported file extensions are ".onnx".

    Returns:
        A list of file paths of the model files.
    """
    models_path = os.path.join(scripts.basedir(), EXTENSION_PATH, "models", "*")
    models = glob.glob(models_path)

    # Add an additional models directory and find files in it
    models_path = os.path.join(scripts.basedir(), "models", "faceswaplab", "*")
    models += glob.glob(models_path)

    # Filter the list to include only files with the supported extensions
    models = [x for x in models if x.endswith(".onnx")]

    return models


def get_current_model() -> str:
    model = opts.data.get("faceswaplab_model", None)
    if model is None:
        models = get_models()
        model = models[0] if len(models) else None
    logger.info("Try to use model : %s", model)
    if not os.path.isfile(model):
        logger.error("The model %s cannot be found or loaded", model)
        raise FileNotFoundError(
            "No faceswap model found. Please add it to the faceswaplab directory."
        )
    return model
