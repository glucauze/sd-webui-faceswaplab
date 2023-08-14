import glob
import os
from typing import List
import modules.scripts as scripts
from modules import scripts
from scripts.faceswaplab_globals import EXPECTED_INSWAPPER_SHA1, EXTENSION_PATH
from modules.shared import opts
from scripts.faceswaplab_utils.faceswaplab_logging import logger
import traceback
import hashlib


def is_sha1_matching(file_path: str, expected_sha1: str) -> bool:
    sha1_hash = hashlib.sha1(usedforsecurity=False)
    try:
        with open(file_path, "rb") as file:
            for byte_block in iter(lambda: file.read(4096), b""):
                sha1_hash.update(byte_block)
            if sha1_hash.hexdigest() == expected_sha1:
                return True
            else:
                return False
    except Exception as e:
        logger.error(
            "Failed to check model hash, check the model is valid or has been downloaded adequately : %e",
            e,
        )
        traceback.print_exc()
        return False


def check_model() -> bool:
    model_path = get_current_swap_model()
    if not is_sha1_matching(
        file_path=model_path, expected_sha1=EXPECTED_INSWAPPER_SHA1
    ):
        logger.error(
            "Suspicious sha1 for model %s, check the model is valid or has been downloaded adequately. Should be %s",
            model_path,
            EXPECTED_INSWAPPER_SHA1,
        )
        return False
    return True


def get_swap_models() -> List[str]:
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


def get_current_swap_model() -> str:
    model = opts.data.get("faceswaplab_model", None)  # type: ignore
    if model is None:
        models = get_swap_models()
        model = models[0] if len(models) else None
    logger.info("Try to use model : %s", model)
    if not os.path.isfile(model):  # type: ignore
        logger.error("The model %s cannot be found or loaded", model)
        raise FileNotFoundError(
            "No faceswap model found. Please add it to the faceswaplab directory."
        )
    return model
