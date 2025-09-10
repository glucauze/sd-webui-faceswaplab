import copy
import os
from dataclasses import dataclass
from pprint import pformat
import traceback
from typing import Any, Dict, Generator, List, Set, Tuple, Optional, Union
import tempfile
from tqdm import tqdm
import sys
from io import StringIO
from contextlib import contextmanager

import cv2
import insightface
import numpy as np
from insightface.app.common import Face as ISFace

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from scripts.faceswaplab_swapping import upscaled_inswapper
from scripts.faceswaplab_swapping.upcaled_inswapper_options import InswappperOptions
from scripts.faceswaplab_utils.imgutils import (
    pil_to_cv2,
)
from scripts.faceswaplab_utils.faceswaplab_logging import logger, save_img_debug
from scripts import faceswaplab_globals
from functools import lru_cache
from scripts.faceswaplab_ui.faceswaplab_unit_settings import FaceSwapUnitSettings
from scripts.faceswaplab_postprocessing.postprocessing import enhance_image
from scripts.faceswaplab_postprocessing.postprocessing_options import (
    PostProcessingOptions,
)
from scripts.faceswaplab_utils.models_utils import get_current_swap_model
from scripts.faceswaplab_utils.typing import CV2ImgU8, Gender, PILImage, Face
from scripts.faceswaplab_inpainting.i2i_pp import img2img_diffusion
from modules import shared
import onnxruntime
from scripts.faceswaplab_utils.sd_utils import get_sd_option


def use_gpu() -> bool:
    return (
        getattr(shared.cmd_opts, "faceswaplab_gpu", False)
        or get_sd_option("faceswaplab_use_gpu", False)
    ) and sys.platform != "darwin"


@lru_cache
def force_install_gpu_providers() -> None:
    # Ugly Ugly hack due to SDNEXT :
    try:
        from scripts.faceswaplab_utils.install_utils import check_install

        logger.warning("Try to reinstall gpu dependencies")
        check_install()
        logger.warning("IF onnxruntime-gpu has been installed successfully, RESTART")
        logger.warning(
            "On SD.NEXT/vladmantic you will also need to check numpy>=1.24.2 and tensorflow>=2.13.0"
        )
    except:
        logger.error(
            "Reinstall has failed (which is normal on windows), please install requirements-gpu.txt manually to enable gpu."
        )


def get_providers() -> List[str]:
    providers = ["CPUExecutionProvider"]
    if use_gpu():
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["CUDAExecutionProvider"]
        else:
            logger.error(
                "CUDAExecutionProvider not found in onnxruntime.available_providers : %s, use CPU instead. Check onnxruntime-gpu is installed.",
                onnxruntime.get_available_providers(),
            )
            force_install_gpu_providers()

    return providers


def is_cpu_provider() -> bool:
    return get_providers() == ["CPUExecutionProvider"]


def cosine_similarity_face(face1: Face, face2: Face) -> float:
    """
    Calculates the cosine similarity between two face embeddings.

    Args:
        face1 (Face): The first face object containing an embedding.
        face2 (Face): The second face object containing an embedding.

    Returns:
        float: The cosine similarity between the face embeddings.

    Note:
        The cosine similarity ranges from 0 to 1, where 1 indicates identical embeddings and 0 indicates completely
        dissimilar embeddings. In this implementation, the similarity is clamped to a minimum value of 0 to ensure a
        non-negative similarity score.
    """
    # Reshape the face embeddings to have a shape of (1, -1)
    assert face1.normed_embedding is not None and face2.normed_embedding is not None
    vec1 = face1.normed_embedding.reshape(1, -1)
    vec2 = face2.normed_embedding.reshape(1, -1)

    # Calculate the cosine similarity between the reshaped embeddings
    similarity = cosine_similarity(vec1, vec2)

    # Return the maximum of 0 and the calculated similarity as the final similarity score
    return max(0, similarity[0, 0])


def compare_faces(img1: PILImage, img2: PILImage) -> float:
    """
    Compares the similarity between two faces extracted from images using cosine similarity.

    Args:
        img1: The first image containing a face.
        img2: The second image containing a face.

    Returns:
        A float value representing the similarity between the two faces (0 to 1).
        Returns -1 if one or both of the images do not contain any faces.
    """

    # Extract faces from the images
    face1 = get_or_default(get_faces(pil_to_cv2(img1)), 0, None)
    face2 = get_or_default(get_faces(pil_to_cv2(img2)), 0, None)

    # Check if both faces are detected
    if face1 is not None and face2 is not None:
        # Calculate the cosine similarity between the faces
        return cosine_similarity_face(face1, face2)

    # Return -1 if one or both of the images do not contain any faces
    return -1


def batch_process(
    src_images: List[Union[PILImage, str]],  # image or filename
    save_path: Optional[str],
    units: List[FaceSwapUnitSettings],
    postprocess_options: Optional[PostProcessingOptions],
) -> Optional[List[PILImage]]:
    """
    Process a batch of images, apply face swapping according to the given settings, and optionally save the resulting images to a specified path.

    Args:
        src_images (List[Union[PILImage, str]]): List of source PIL Images to process or list of images file names
        save_path (Optional[str]): Destination path where the processed images will be saved. If None, no images are saved.
        units (List[FaceSwapUnitSettings]): List of FaceSwapUnitSettings to apply to the images.
        postprocess_options (PostProcessingOptions): Post-processing settings to be applied to the images.

    Returns:
        Optional[List[PILImage]]: List of processed images, or None in case of an exception.

    Raises:
        Any exceptions raised by the underlying process will be logged and the function will return None.
    """
    try:
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        units = [u for u in units if u.enable]
        if src_images is not None and len(units) > 0:
            result_images = []
            for src_image in src_images:
                path: str = ""
                if isinstance(src_image, str):
                    if save_path:
                        path = os.path.join(
                            save_path, "swapped_" + os.path.basename(src_image)
                        )
                    src_image = Image.open(src_image)
                elif save_path:
                    path = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".png", dir=save_path
                    ).name
                assert isinstance(src_image, Image.Image)

                current_images = []
                swapped_images = process_images_units(
                    get_current_swap_model(), images=[(src_image, None)], units=units
                )
                if swapped_images and len(swapped_images) > 0:
                    current_images += [img for img, _ in swapped_images]

                logger.info("%s images generated", len(current_images))

                if postprocess_options:
                    for i, img in enumerate(current_images):
                        current_images[i] = enhance_image(img, postprocess_options)

                if save_path:
                    for img in current_images:
                        img.save(path)

                result_images += current_images
            return result_images
    except Exception as e:
        logger.error("Batch Process error : %s", e)
        import traceback

        traceback.print_exc()
    return None


def extract_faces(
    images: List[PILImage],
    extract_path: Optional[str],
    postprocess_options: PostProcessingOptions,
) -> Optional[List[PILImage]]:
    """
    Extracts faces from a list of image files.

    Given a list of image file paths, this function opens each image, extracts the faces,
    and saves them in a specified directory. Post-processing is applied to each extracted face,
    and the processed faces are saved as separate PNG files.

    Parameters:
    files (Optional[List[Image]]): List of file paths to the images to extract faces from.
    extract_path (Optional[str]): Path where the extracted faces will be saved.
                                  If no path is provided, a temporary directory will be created.
    postprocess_options (PostProcessingOptions): Post-processing settings to be applied to the images.

    Returns:
    Optional[List[img]]: List of face images
    """

    try:
        if extract_path:
            os.makedirs(extract_path, exist_ok=True)

        if images:
            result_images: list[PILImage] = []
            for img in images:
                faces = get_faces(pil_to_cv2(img))

                if faces:
                    face_images = []
                    for face in faces:
                        bbox = face.bbox.astype(int)  # type: ignore
                        x_min, y_min, x_max, y_max = bbox
                        face_image = img.crop((x_min, y_min, x_max, y_max))

                        if postprocess_options and (
                            postprocess_options.face_restorer_name
                            or postprocess_options.restorer_visibility
                        ):
                            postprocess_options.scale = (
                                1 if face_image.width > 512 else 512 // face_image.width
                            )
                            face_image = enhance_image(face_image, postprocess_options)

                        if extract_path:
                            path = tempfile.NamedTemporaryFile(
                                delete=False, suffix=".png", dir=extract_path
                            ).name
                            face_image.save(path)
                        face_images.append(face_image)

                    result_images += face_images

            return result_images
    except Exception as e:
        logger.error("Failed to extract : %s", e)
        import traceback

        traceback.print_exc()
    return None


class FaceModelException(Exception):
    """Exception raised when an error is encountered in the face model."""

    def __init__(self, message: str) -> None:
        """
        Args:
            message: A string containing the error description.
        """
        self.message = message
        super().__init__(self.message)


@contextmanager
def capture_stdout() -> Generator[StringIO, None, None]:
    """
    Capture and yield the printed messages to stdout.

    This context manager temporarily replaces sys.stdout with a StringIO object,
    capturing all printed output. After the context block is exited, sys.stdout
    is restored to its original value.

    Example usage:
    with capture_stdout() as captured:
        print("Hello, World!")
        output = captured.getvalue()
        # output now contains "Hello, World!\n"

    Returns:
        A StringIO object containing the captured output.
    """
    original_stdout = sys.stdout  # Type: ignore
    captured_stdout = StringIO()
    sys.stdout = captured_stdout  # Type: ignore
    try:
        yield captured_stdout
    finally:
        sys.stdout = original_stdout  # Type: ignore


@lru_cache(maxsize=3)
def getAnalysisModel(
    det_size: Tuple[int, int] = (640, 640),
    det_thresh: float = 0.5,
    use_gpu: bool = False,
) -> insightface.app.FaceAnalysis:
    """
    Retrieves the analysis model for face analysis.

    Returns:
        insightface.app.FaceAnalysis: The analysis model for face analysis.
    """
    try:
        if not os.path.exists(faceswaplab_globals.ANALYZER_DIR):
            os.makedirs(faceswaplab_globals.ANALYZER_DIR)

        providers = get_providers()
        logger.info(
            f"Load analysis model det_size={det_size}, det_thresh={det_thresh}, providers = {providers}, will take some time. (> 30s)"
        )
        # Initialize the analysis model with the specified name and providers

        with tqdm(
            total=1,
            desc=f"Loading {det_size} analysis model (first time is slow)",
            unit="model",
        ) as pbar:
            with capture_stdout() as captured:
                model = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    providers=providers,
                    root=faceswaplab_globals.ANALYZER_DIR,
                )

                # Prepare the analysis model for face detection with the specified detection size
                model.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
            pbar.update(1)
        logger.info("%s", pformat(captured.getvalue()))

        return model
    except Exception as e:
        logger.error(
            "Loading of swapping model failed, please check the requirements (On Windows, download and install Visual Studio. During the install, make sure to include the Python and C++ packages.)"
        )
        raise FaceModelException("Loading of analysis model failed")


@lru_cache(maxsize=1)
def getFaceSwapModel(
    model_path: str, use_gpu: bool = False
) -> upscaled_inswapper.UpscaledINSwapper:
    """
    Retrieves the face swap model and initializes it if necessary.

    Args:
        model_path (str): Path to the face swap model.

    Returns:
        insightface.model_zoo.FaceModel: The face swap model.
    """
    try:
        providers = get_providers()
        with tqdm(total=1, desc="Loading swap model", unit="model") as pbar:
            with capture_stdout() as captured:
                model = upscaled_inswapper.UpscaledINSwapper(
                    insightface.model_zoo.get_model(model_path, providers=providers)  # type: ignore
                )
            pbar.update(1)
        logger.info("%s", pformat(captured.getvalue()))
        return model

    except Exception as e:
        logger.error(
            "Loading of swapping model failed, please check the requirements (On Windows, download and install Visual Studio. During the install, make sure to include the Python and C++ packages.)"
        )
        traceback.print_exc()
        raise FaceModelException("Loading of swapping model failed")


def get_faces(
    img_data: CV2ImgU8,
    det_thresh: Optional[float] = None,
    det_size: Tuple[int, int] = (640, 640),
) -> List[Face]:
    """
    Detects and retrieves faces from an image using an analysis model.

    Args:
        img_data (CV2ImgU8): The image data as a NumPy array.
        det_size (tuple): The desired detection size (width, height). Defaults to (640, 640).
        sort_by_face_size (bool) : Will sort the faces by their size from larger to smaller face

    Returns:
        list: A list of detected faces, sorted by their x-coordinate of the bounding box.
    """

    if det_thresh is None:
        det_thresh = get_sd_option("faceswaplab_detection_threshold", 0.5)

    auto_det_size = get_sd_option("faceswaplab_auto_det_size", True)
    if not auto_det_size:
        x = get_sd_option("faceswaplab_det_size", 640)
        det_size = (x, x)

    face_analyser = getAnalysisModel(
        det_size=det_size, det_thresh=det_thresh, use_gpu=not is_cpu_provider()
    )

    # Get the detected faces from the image using the analysis model
    faces = face_analyser.get(img_data)

    # If no faces are detected and the detection size is larger than 320x320,
    # recursively call the function with a smaller detection size
    if len(faces) == 0:
        if auto_det_size:
            if det_size[0] > 320 and det_size[1] > 320:
                det_size_half = (det_size[0] // 2, det_size[1] // 2)
                return get_faces(
                    img_data, det_size=det_size_half, det_thresh=det_thresh
                )

        # If no faces are detected print a warning to user about change in detection
        else:
            if det_size[0] > 320:
                logger.warning(
                    "No faces detected, you might want to play with det_size by reducing it (in sd global settings). Lower (320) means more detection but less precise. Or activate auto-det-size."
                )

    try:
        # Sort the detected faces based on their x-coordinate of the bounding box
        return sorted(faces, key=lambda x: x.bbox[0])  # type: ignore
    except Exception as e:
        logger.error("Failed to get faces %s", e)
        traceback.print_exc()
        return []


@dataclass
class FaceFilteringOptions:
    faces_index: Set[int]
    source_gender: Optional[int] = None  # if none will not use same gender
    sort_by_face_size: bool = False


def filter_faces(
    all_faces: List[Face], filtering_options: FaceFilteringOptions
) -> List[Face]:
    """
    Sorts and filters a list of faces based on specified criteria.

    This function takes a list of Face objects and can sort them by face size and filter them by gender.
    Sorting by face size is performed if sort_by_face_size is set to True, and filtering by gender is
    performed if source_gender is provided.

    :param faces: A list of Face objects representing the faces to be sorted and filtered.
    :param faces_index: A set of faces index
    :param source_gender: An optional integer representing the gender by which to filter the faces.
                          If provided, only faces with the specified gender will be included in the result.
    :param sort_by_face_size: A boolean indicating whether to sort the faces by size. If True, faces are
                              sorted in descending order by size, calculated as the area of the bounding box.
    :return: A list of Face objects sorted and filtered according to the specified criteria.
    """
    filtered_faces = copy.copy(all_faces)
    if filtering_options.sort_by_face_size:
        filtered_faces = sorted(
            all_faces,
            reverse=True,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),  # type: ignore
        )

    if filtering_options.source_gender is not None:
        filtered_faces = [
            face
            for face in filtered_faces
            if face["gender"] == filtering_options.source_gender
        ]
    return [
        face
        for i, face in enumerate(filtered_faces)
        if i in filtering_options.faces_index
    ]


@dataclass
class ImageResult:
    """
    Represents the result of an image swap operation
    """

    image: PILImage
    """
    The image object with the swapped face
    """

    similarity: Dict[int, float]
    """
    A dictionary mapping face indices to their similarity scores.
    The similarity scores are represented as floating-point values between 0 and 1.
    """

    ref_similarity: Dict[int, float]
    """
    A dictionary mapping face indices to their similarity scores compared to a reference image.
    The similarity scores are represented as floating-point values between 0 and 1.
    """


def get_or_default(l: List[Any], index: int, default: Any) -> Any:
    """
    Retrieve the value at the specified index from the given list.
    If the index is out of bounds, return the default value instead.

    Args:
        l (list): The input list.
        index (int): The index to retrieve the value from.
        default: The default value to return if the index is out of bounds.

    Returns:
        The value at the specified index if it exists, otherwise the default value.
    """
    return l[index] if index < len(l) else default


def get_faces_from_img_files(images: List[PILImage]) -> List[Face]:
    """
    Extracts faces from a list of image files.

    Args:
        images (list): A list of PILImage objects representing image files.

    Returns:
        list: A list of detected faces.

    """

    faces: List[Face] = []

    if len(images) > 0:
        for img in images:
            face = get_or_default(
                get_faces(pil_to_cv2(img)), 0, None
            )  # Extract faces from the image
            if face is not None:
                faces.append(face)  # Add the detected face to the list of faces

    return faces


def blend_faces(faces: List[Face], gender: Gender = Gender.AUTO) -> Optional[Face]:
    """
    Blends the embeddings of multiple faces into a single face.

    Args:
        faces (List[Face]): List of Face objects.

    Returns:
        Face: The blended Face object with the averaged embedding.
              Returns None if the input list is empty.

    Raises:
        ValueError: If the embeddings have different shapes.

    """
    embeddings: list[Any] = [face.embedding for face in faces]

    if len(embeddings) > 0:
        embedding_shape = embeddings[0].shape

        # Check if all embeddings have the same shape
        for embedding in embeddings:
            if embedding.shape != embedding_shape:
                raise ValueError("embedding shape mismatch")

        # Compute the mean of all embeddings
        blended_embedding = np.mean(embeddings, axis=0)

        if gender == Gender.AUTO:
            int_gender: int = faces[0].gender  # type: ignore
        else:
            int_gender: int = gender.value

        assert -1 < int_gender < 2, "wrong gender"

        logger.info("Int Gender : %s", int_gender)

        # Create a new Face object using the properties of the first face in the list
        # Assign the blended embedding to the blended Face object
        blended = ISFace(
            embedding=blended_embedding, gender=int_gender, age=faces[0].age
        )

        return blended

    # Return None if the input list is empty
    return None


def swap_face(
    source_face: Face,
    target_img: PILImage,
    target_faces: List[Face],
    model: str,
    swapping_options: Optional[InswappperOptions],
) -> ImageResult:
    """
    Swaps faces in the target image with the source face.

    Args:
        source_face (CV2ImgU8): The source face to be swapped.
        target_img (PILImage): The target image to swap faces in.
        model (str): Path to the face swap model.

    Returns:
        ImageResult: An object containing the swapped image and similarity scores.

    """
    return_result = ImageResult(target_img, {}, {})
    target_img_cv2: CV2ImgU8 = cv2.cvtColor(
        np.array(target_img), cv2.COLOR_RGB2BGR
    ).astype("uint8")
    try:
        gender = source_face["gender"]
        logger.info("Source Gender %s", gender)
        if source_face is not None:
            result: CV2ImgU8 = target_img_cv2
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
            face_swapper = getFaceSwapModel(model_path, use_gpu=not is_cpu_provider())
            logger.info("Target faces count : %s", len(target_faces))

            for i, swapped_face in enumerate(target_faces):
                logger.info(f"swap face {i}")

                result = face_swapper.get(
                    img=result,
                    target_face=swapped_face,
                    source_face=source_face,
                    options=swapping_options,
                )  # type: ignore

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            return_result.image = result_image

    except Exception as e:
        logger.error("Conversion failed %s", e)
        raise e
    return return_result


def compute_similarity(
    reference_face: Face,
    source_face: Face,
    swapped_image: PILImage,
    filtering: FaceFilteringOptions,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    similarity: Dict[int, float] = {}
    ref_similarity: Dict[int, float] = {}
    try:
        swapped_image_cv2: CV2ImgU8 = cv2.cvtColor(
            np.array(swapped_image), cv2.COLOR_RGB2BGR
        )
        new_faces = filter_faces(get_faces(swapped_image_cv2), filtering)
        if len(new_faces) == 0:
            logger.error("compute_similarity : No faces to compare with !")
            return None

        for i, swapped_face in enumerate(new_faces):
            logger.info(f"compare face {i}")
            similarity[i] = cosine_similarity_face(source_face, swapped_face)
            ref_similarity[i] = cosine_similarity_face(reference_face, swapped_face)

            logger.info(f"similarity {similarity}")
            logger.info(f"ref similarity {ref_similarity}")

        return (similarity, ref_similarity)
    except Exception as e:
        logger.error("Similarity processing failed %s", e)
        raise e
    return None


def process_image_unit(
    model: str,
    unit: FaceSwapUnitSettings,
    image: PILImage,
    info: Optional[str] = None,
    force_blend: bool = False,
) -> List[Tuple[PILImage, Optional[str]]]:
    """Process one image and return a List of (image, info) (one if blended, many if not).

    Args:
        unit : the current unit
        image : the image where to apply swapping
        info : The info

    Returns:
        List of tuple of (image, info) where image is the image where swapping has been applied and info is the image info with similarity infos.
    """

    results = []
    if unit.enable:
        faces = get_faces(pil_to_cv2(image))

        if not unit.blend_faces and not force_blend:
            src_faces = unit.faces
            logger.info(f"will generate {len(src_faces)} images")
        else:
            logger.info("blend all faces together")
            src_faces = [unit.blended_faces]

        for i, src_face in enumerate(src_faces):
            current_image = image

            logger.info(f"Process face {i}")
            if unit.reference_face is not None:
                reference_face = unit.reference_face
            else:
                logger.info("Use source face as reference face")
                reference_face = src_face

            face_filtering_options = FaceFilteringOptions(
                faces_index=unit.faces_index,
                source_gender=src_face["gender"] if unit.same_gender else None,
                sort_by_face_size=unit.sort_by_size,
            )

            target_faces: List[Face] = filter_faces(
                all_faces=faces, filtering_options=face_filtering_options
            )

            # Apply pre-inpainting to image
            if unit.pre_inpainting.inpainting_denoising_strengh > 0:
                current_image = img2img_diffusion(
                    img=current_image, faces=target_faces, options=unit.pre_inpainting
                )

            save_img_debug(image, "Before swap")
            result: ImageResult = swap_face(
                source_face=src_face,
                target_img=current_image,
                target_faces=target_faces,
                model=model,
                swapping_options=unit.swapping_options,
            )
            # Apply post-inpainting to image
            if unit.post_inpainting.inpainting_denoising_strengh > 0:
                result.image = img2img_diffusion(
                    img=result.image, faces=target_faces, options=unit.post_inpainting
                )

            save_img_debug(result.image, "After swap")

            if unit.compute_similarity:
                similarities = compute_similarity(
                    reference_face=reference_face,
                    source_face=src_face,
                    swapped_image=result.image,
                    filtering=face_filtering_options,
                )
                if similarities:
                    (result.similarity, result.ref_similarity) = similarities
                else:
                    logger.error("Failed to compute similarity")

            if result.image is None:
                logger.error("Result image is None")
            if (
                (not unit.check_similarity)
                or result.similarity
                and all(
                    [result.similarity.values() != 0]
                    + [x >= unit.min_sim for x in result.similarity.values()]
                )
                and all(
                    [result.ref_similarity.values() != 0]
                    + [x >= unit.min_ref_sim for x in result.ref_similarity.values()]
                )
            ):
                results.append(
                    (
                        result.image,
                        f"{info}, similarity = {result.similarity}, ref_similarity = {result.ref_similarity}",
                    )
                )
            else:
                logger.warning(
                    f"skip, similarity to low, sim = {result.similarity} (target {unit.min_sim}) ref sim = {result.ref_similarity} (target = {unit.min_ref_sim})"
                )
    logger.debug("process_image_unit : Unit produced %s results", len(results))
    return results


def process_images_units(
    model: str,
    units: List[FaceSwapUnitSettings],
    images: List[Tuple[Optional[PILImage], Optional[str]]],
    force_blend: bool = False,
) -> Optional[List[Tuple[PILImage, str]]]:
    """
    Process a list of images using a specified model and unit settings for face swapping.

    Args:
        model (str): The name of the model to use for processing.
        units (List[FaceSwapUnitSettings]): A list of settings for face swap units to apply on each image.
        images (List[Tuple[Optional[PILImage], Optional[str]]]): A list of tuples, each containing
            an image and its associated info string. If an image or info string is not available,
            its value can be None.
        upscaled_swapper (bool, optional): If True, uses an upscaled version of the face swapper.
            Defaults to False.
        force_blend (bool, optional): If True, forces the blending of the swapped face on the original
            image. Defaults to False.

    Returns:
        Optional[List[Tuple[PILImage, str]]]: A list of tuples, each containing a processed image
            and its associated info string. If no units are provided for processing, returns None.

    """
    if len(units) == 0:
        logger.info("Finished processing image, return %s images", len(images))
        return None

    logger.debug("%s more units", len(units))

    processed_images = []
    for i, (image, info) in enumerate(images):
        logger.debug("Processing image %s", i)
        swapped = process_image_unit(model, units[0], image, info, force_blend)
        logger.debug("Image %s -> %s images", i, len(swapped))
        nexts = process_images_units(model, units[1:], swapped, force_blend)
        if nexts:
            processed_images.extend(nexts)
        else:
            processed_images.extend(swapped)

    return processed_images
