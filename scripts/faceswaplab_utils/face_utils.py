import glob
import os
from typing import List
from insightface.app.common import Face
from safetensors.torch import save_file, safe_open
import torch

import modules.scripts as scripts
from modules import scripts
from scripts.faceswaplab_utils.faceswaplab_logging import logger


def save_face(face: Face, filename: str) -> None:
    tensors = {
        "embedding": torch.tensor(face["embedding"]),
        "gender": torch.tensor(face["gender"]),
        "age": torch.tensor(face["age"]),
    }
    save_file(tensors, filename)


def load_face(filename: str) -> Face:
    face = {}
    logger.debug("Try to load face from %s", filename)
    with safe_open(filename, framework="pt", device="cpu") as f:
        logger.debug("File contains %s keys", f.keys())
        for k in f.keys():
            logger.debug("load key %s", k)
            face[k] = f.get_tensor(k).numpy()
    logger.debug("face : %s", face)
    return Face(face)


def get_face_checkpoints() -> List[str]:
    """
    Retrieve a list of face checkpoint paths.

    This function searches for face files with the extension ".safetensors" in the specified directory and returns a list
    containing the paths of those files.

    Returns:
        list: A list of face paths, including the string "None" as the first element.
    """
    faces_path = os.path.join(
        scripts.basedir(), "models", "faceswaplab", "faces", "*.safetensors"
    )
    faces = glob.glob(faces_path)
    return ["None"] + faces
