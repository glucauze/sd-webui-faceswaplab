import glob
import os
from typing import List
from insightface.app.common import Face
from safetensors.torch import save_file, safe_open
import torch

import modules.scripts as scripts
from modules import scripts
from scripts.faceswaplab_utils.faceswaplab_logging import logger
import dill as pickle  # will be removed in future versions


def save_face(face: Face, filename: str) -> None:
    tensors = {
        "embedding": torch.tensor(face["embedding"]),
        "gender": torch.tensor(face["gender"]),
        "age": torch.tensor(face["age"]),
    }
    save_file(tensors, filename)


def load_face(filename: str) -> Face:
    if filename.endswith(".pkl"):
        logger.warning(
            "Pkl files for faces are deprecated to enhance safety, they will be unsupported in future versions."
        )
        logger.warning("The file will be converted to .safetensors")
        logger.warning(
            "You can also use this script https://gist.github.com/glucauze/4a3c458541f2278ad801f6625e5b9d3d"
        )
        with open(filename, "rb") as file:
            logger.info("Load pkl")
            face = Face(pickle.load(file))
            logger.warning(
                "Convert to safetensors, you can remove the pkl version once you have ensured that the safetensor is working"
            )
            save_face(face, filename.replace(".pkl", ".safetensors"))
        return face

    elif filename.endswith(".safetensors"):
        face = {}
        with safe_open(filename, framework="pt", device="cpu") as f:
            for k in f.keys():
                logger.debug("load key %s", k)
                face[k] = f.get_tensor(k).numpy()
        return Face(face)

    raise NotImplementedError("Unknown file type, face extraction not implemented")


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

    faces_path = os.path.join(
        scripts.basedir(), "models", "faceswaplab", "faces", "*.pkl"
    )
    faces += glob.glob(faces_path)

    return ["None"] + sorted(faces)
