import glob
import os
from typing import *
from insightface.app.common import Face
from safetensors.torch import save_file, safe_open
import torch

import modules.scripts as scripts
from modules import scripts
from scripts.faceswaplab_swapping.upcaled_inswapper_options import InswappperOptions
from scripts.faceswaplab_utils.faceswaplab_logging import logger
from scripts.faceswaplab_utils.typing import *
from scripts.faceswaplab_utils import imgutils
from scripts.faceswaplab_utils.models_utils import get_swap_models
import traceback

from scripts.faceswaplab_swapping import swapper
from pprint import pformat
import re
from client_api import api_utils
import tempfile


def sanitize_name(name: str) -> str:
    """
    Sanitize the input name by removing special characters and replacing spaces with underscores.

    Parameters:
        name (str): The input name to be sanitized.

    Returns:
        str: The sanitized name with special characters removed and spaces replaced by underscores.
    """
    name = re.sub("[^A-Za-z0-9_. ]+", "", name)
    name = name.replace(" ", "_")
    return name[:255]


def build_face_checkpoint_and_save(
    images: List[PILImage],
    name: str,
    gender: Gender = Gender.AUTO,
    overwrite: bool = False,
    path: Optional[str] = None,
) -> Optional[PILImage]:
    """
    Builds a face checkpoint using the provided image files, performs face swapping,
    and saves the result to a file. If a blended face is successfully obtained and the face swapping
    process succeeds, the resulting image is returned. Otherwise, None is returned.

    Args:
        batch_files (list): List of image file paths used to create the face checkpoint.
        name (str): The name assigned to the face checkpoint.

    Returns:
        PIL.PILImage or None: The resulting swapped face image if the process is successful; None otherwise.
    """

    try:
        name = sanitize_name(name)
        images = images or []
        logger.info("Build %s with %s images", name, len(images))
        faces: List[Face] = swapper.get_faces_from_img_files(images=images)
        if faces is None or len(faces) == 0:
            logger.error("No source faces found")
            return None

        blended_face: Optional[Face] = swapper.blend_faces(faces, gender=gender)
        preview_path = os.path.join(
            scripts.basedir(), "extensions", "sd-webui-faceswaplab", "references"
        )

        reference_preview_img: PILImage
        if blended_face:
            if blended_face["gender"] == 0:
                reference_preview_img = Image.open(
                    os.path.join(preview_path, "woman.png")
                )
            else:
                reference_preview_img = Image.open(
                    os.path.join(preview_path, "man.png")
                )

            if name == "":
                name = "default_name"
            logger.debug("Face %s", pformat(blended_face))
            target_face = swapper.get_or_default(
                swapper.get_faces(imgutils.pil_to_cv2(reference_preview_img)), 0, None
            )
            if target_face is None:
                logger.error(
                    "Failed to open reference image, cannot create preview : That should not happen unless you deleted the references folder or change the detection threshold."
                )
            else:
                result: swapper.ImageResult = swapper.swap_face(
                    target_faces=[target_face],
                    source_face=blended_face,
                    target_img=reference_preview_img,
                    model=get_swap_models()[0],
                    swapping_options=InswappperOptions(
                        face_restorer_name="CodeFormer",
                        restorer_visibility=1,
                        upscaler_name="Lanczos",
                        codeformer_weight=1,
                        improved_mask=True,
                        color_corrections=False,
                        sharpen=True,
                    ),
                )
                preview_image = result.image

                if path:
                    file_path = path
                else:
                    file_path = os.path.join(
                        get_checkpoint_path(), f"{name}.safetensors"
                    )
                    if not overwrite:
                        file_number = 1
                        while os.path.exists(file_path):
                            file_path = os.path.join(
                                get_checkpoint_path(),
                                f"{name}_{file_number}.safetensors",
                            )
                            file_number += 1
                save_face(filename=file_path, face=blended_face)
                preview_image.save(file_path + ".png")
                try:
                    data = load_face(file_path)
                    logger.debug(data)
                except Exception as e:
                    logger.error("Error loading checkpoint, after creation %s", e)
                    traceback.print_exc()

                return preview_image

        else:
            logger.error("No face found")
            return None  # type: ignore
    except Exception as e:
        logger.error("Failed to build checkpoint %s", e)
        traceback.print_exc()
        return None


def save_face(face: Face, filename: str) -> None:
    try:
        tensors = {
            "embedding": torch.tensor(face["embedding"]),
            "gender": torch.tensor(face["gender"]),
            "age": torch.tensor(face["age"]),
        }
        save_file(tensors, filename)
    except Exception as e:
        traceback.print_exc
        logger.error("Failed to save checkpoint %s", e)
        raise e


def load_face(name: str) -> Optional[Face]:
    if name.startswith("data:application/face;base64,"):
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            api_utils.base64_to_safetensors(name, temp_file.name)
            face = {}
            with safe_open(temp_file.name, framework="pt", device="cpu") as f:
                for k in f.keys():
                    logger.debug("load key %s", k)
                    face[k] = f.get_tensor(k).numpy()
            return Face(face)

    filename = matching_checkpoint(name)
    if filename is None:
        return None

    if filename.endswith(".pkl"):
        logger.warning(
            "Pkl files for faces are deprecated to enhance safety, you need to convert them"
        )
        logger.warning("The file will be converted to .safetensors")
        logger.warning(
            "You can also use this script https://gist.github.com/glucauze/4a3c458541f2278ad801f6625e5b9d3d"
        )
        return None

    elif filename.endswith(".safetensors"):
        face = {}
        with safe_open(filename, framework="pt", device="cpu") as f:
            for k in f.keys():
                logger.debug("load key %s", k)
                face[k] = f.get_tensor(k).numpy()
        return Face(face)

    raise NotImplementedError("Unknown file type, face extraction not implemented")


def get_checkpoint_path() -> str:
    checkpoint_path = os.path.join(scripts.basedir(), "models", "faceswaplab", "faces")
    os.makedirs(checkpoint_path, exist_ok=True)
    return checkpoint_path


def matching_checkpoint(name: str) -> Optional[str]:
    """
    Retrieve the full path of a checkpoint file matching the given name.

    If the name already includes a path separator, it is returned as-is. Otherwise, the function looks for a matching
    file with the extensions ".safetensors" or ".pkl" in the checkpoint directory.

    Args:
        name (str): The name or path of the checkpoint file.

    Returns:
        Optional[str]: The full path of the matching checkpoint file, or None if no match is found.
    """

    # If the name already includes a path separator, return it as is
    if os.path.sep in name:
        return name

    # If the name doesn't end with the specified extensions, look for a matching file
    if not (name.endswith(".safetensors") or name.endswith(".pkl")):
        # Try appending each extension and check if the file exists in the checkpoint path
        for ext in [".safetensors", ".pkl"]:
            full_path = os.path.join(get_checkpoint_path(), name + ext)
            if os.path.exists(full_path):
                return full_path
        # If no matching file is found, return None
        return None

    # If the name already ends with the specified extensions, simply complete the path
    return os.path.join(get_checkpoint_path(), name)


def get_face_checkpoints() -> List[str]:
    """
    Retrieve a list of face checkpoint paths.

    This function searches for face files with the extension ".safetensors" in the specified directory and returns a list
    containing the paths of those files.

    Returns:
        list: A list of face paths, including the string "None" as the first element.
    """
    faces_path = os.path.join(get_checkpoint_path(), "*.safetensors")
    faces = glob.glob(faces_path)

    faces_path = os.path.join(get_checkpoint_path(), "*.pkl")
    faces += glob.glob(faces_path)

    return ["None"] + [os.path.basename(face) for face in sorted(faces)]
