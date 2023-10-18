import traceback
from pprint import pformat
from typing import *
from scripts.faceswaplab_swapping import face_checkpoints
from scripts.faceswaplab_utils.sd_utils import get_sd_option
from scripts.faceswaplab_utils.typing import *
import gradio as gr
import onnx
import pandas as pd
from PIL import Image

import scripts.faceswaplab_swapping.swapper as swapper
from scripts.faceswaplab_postprocessing.postprocessing_options import (
    PostProcessingOptions,
)
from scripts.faceswaplab_ui.faceswaplab_postprocessing_ui import postprocessing_ui
from scripts.faceswaplab_ui.faceswaplab_unit_settings import FaceSwapUnitSettings
from scripts.faceswaplab_ui.faceswaplab_unit_ui import faceswap_unit_ui
from scripts.faceswaplab_utils import imgutils
from scripts.faceswaplab_utils.faceswaplab_logging import logger
from scripts.faceswaplab_utils.models_utils import get_swap_models
from scripts.faceswaplab_utils.ui_utils import dataclasses_from_flat_list


def compare(img1: PILImage, img2: PILImage) -> str:
    """
    Compares the similarity between two faces extracted from images using cosine similarity.

    Args:
        img1: The first image containing a face.
        img2: The second image containing a face.

    Returns:
        A str of a float value representing the similarity between the two faces (0 to 1).
        Returns"You need 2 images to compare" if one or both of the images do not contain any faces.
    """
    try:
        if img1 is not None and img2 is not None:
            return str(swapper.compare_faces(img1, img2))
    except Exception as e:
        logger.error("Fail to compare", e)

        traceback.print_exc()
    return "You need 2 images to compare"


def extract_faces(
    files: List[gr.File],
    extract_path: Optional[str],
    *components: Tuple[gr.components.Component, ...],
) -> Optional[List[PILImage]]:
    """
    Extracts faces from a list of image files.

    Given a list of image file paths, this function opens each image, extracts the faces,
    and saves them in a specified directory. Post-processing is applied to each extracted face,
    and the processed faces are saved as separate PNG files.

    Parameters:
    files (Optional[List[str]]): List of file paths to the images to extract faces from.
    extract_path (Optional[str]): Path where the extracted faces will be saved.
                                  If no path is provided, a temporary directory will be created.
    components (List[gr.components.Component]): List of components for post-processing.

    Returns:
    Optional[List[str]]: List of file paths to the saved images of the extracted faces.
                         If no faces are found, None is returned.
    """

    if files and len(files) == 0:
        logger.error("You need at least one image file to extract")
        return []
    try:
        postprocess_options = dataclasses_from_flat_list(
            [PostProcessingOptions], components
        ).pop()
        images = [
            Image.open(file.name) for file in files  # type: ignore
        ]  # potentially greedy but Image.open is supposed to be lazy
        result_images = swapper.extract_faces(
            images, extract_path=extract_path, postprocess_options=postprocess_options
        )
        return result_images
    except Exception as e:
        logger.error("Failed to extract : %s", e)

        traceback.print_exc()
    return None


def analyse_faces(image: PILImage, det_threshold: float = 0.5) -> Optional[str]:
    """
    Function to analyze the faces in an image and provide a detailed report.

    Parameters
    ----------
    image : PIL.PILImage
        The input image where faces will be detected. The image must be a PIL Image object.

    det_threshold : float, optional
        The detection threshold for the face detection process, by default 0.5. It determines
        the confidence level at which the function will consider a detected object as a face.
        Value should be in the range [0, 1], with higher values indicating greater certainty.

    Returns
    -------
    str or None
        Returns a formatted string providing details about each face detected in the image.
        For each face, the string will include an index and a set of facial details.
        In the event of an exception (e.g., analysis failure), the function will log the error
        and return None.

    Raises
    ------
    This function handles exceptions internally and does not raise.

    Examples
    --------
    >>> image = Image.open("test.jpg")
    >>> print(analyse_faces(image, 0.7))
    """

    try:
        faces = swapper.get_faces(imgutils.pil_to_cv2(image), det_thresh=det_threshold)
        result = ""
        for i, face in enumerate(faces):
            result += f"\nFace {i} \n" + "=" * 40 + "\n"
            result += pformat(face) + "\n"
            result += "=" * 40
        return result if result else None

    except Exception as e:
        logger.error("Analysis Failed : %s", e)

        traceback.print_exc()
    return None


def build_face_checkpoint_and_save(
    batch_files: List[gr.File], name: str, str_gender: str, overwrite: bool
) -> PILImage:
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
        if not batch_files:
            logger.error("No face found")
            return None  # type: ignore (Optional not really supported by old gradio)

        gender = getattr(Gender, str_gender)
        logger.info("Chosen gender : %s", gender)
        images: list[PILImage] = [Image.open(file.name) for file in batch_files]  # type: ignore
        preview_image: PILImage | None = (
            face_checkpoints.build_face_checkpoint_and_save(
                images=images, name=name, overwrite=overwrite, gender=gender
            )
        )
    except Exception as e:
        logger.error("Failed to build checkpoint %s", e)

        traceback.print_exc()
        return None  # type: ignore
    return preview_image  # type: ignore


def explore_onnx_faceswap_model(model_path: str) -> pd.DataFrame:
    try:
        data: Dict[str, Any] = {
            "Node Name": [],
            "Op Type": [],
            "Inputs": [],
            "Outputs": [],
            "Attributes": [],
        }
        if model_path:
            model = onnx.load(model_path)
            for node in model.graph.node:
                data["Node Name"].append(pformat(node.name))
                data["Op Type"].append(pformat(node.op_type))
                data["Inputs"].append(pformat(node.input))
                data["Outputs"].append(pformat(node.output))
                attributes = []
                for attr in node.attribute:
                    attr_name = attr.name
                    attr_value = attr.t
                    attributes.append(
                        "{} = {}".format(pformat(attr_name), pformat(attr_value))
                    )
                data["Attributes"].append(attributes)

        df = pd.DataFrame(data)
    except Exception as e:
        logger.error("Failed to explore model %s", e)

        traceback.print_exc()
        return None  # type: ignore
    return df


def batch_process(
    files: List[gr.File], save_path: str, *components: Tuple[Any, ...]
) -> List[PILImage]:
    try:
        units_count = get_sd_option("faceswaplab_units_count", 3)

        classes: List[Any] = dataclasses_from_flat_list(
            [FaceSwapUnitSettings] * units_count + [PostProcessingOptions],
            components,
        )
        units: List[FaceSwapUnitSettings] = [
            u for u in classes if isinstance(u, FaceSwapUnitSettings)
        ]
        postprocess_options = classes[-1]

        images_paths = [file.name for file in files]  # type: ignore

        return (
            swapper.batch_process(
                images_paths,
                save_path=save_path,
                units=units,
                postprocess_options=postprocess_options,
            )
            or []
        )
    except Exception as e:
        logger.error("Batch Process error : %s", e)

        traceback.print_exc()
    return []


def tools_ui() -> None:
    models = get_swap_models()
    with gr.Tab("Tools"):
        with gr.Tab("Build"):
            gr.Markdown(
                """Build a face based on a batch list of images. Will blend the resulting face and store the checkpoint in the faceswaplab/faces directory."""
            )
            with gr.Row():
                build_batch_files = gr.components.File(
                    type="file",
                    file_count="multiple",
                    label="Batch Sources Images",
                    optional=True,
                    elem_id="faceswaplab_build_batch_files",
                )
                preview = gr.components.Image(
                    type="pil",
                    label="Preview",
                    width=512,
                    height=512,
                    interactive=False,
                    elem_id="faceswaplab_build_preview_face",
                )
            build_name = gr.Textbox(
                value="Face",
                placeholder="Name of the character",
                label="Name of the character",
                elem_id="faceswaplab_build_character_name",
            )
            build_gender = gr.Dropdown(
                value=Gender.AUTO.name,
                choices=[e.name for e in Gender],
                placeholder="Gender of the character",
                label="Gender of the character",
                elem_id="faceswaplab_build_character_gender",
            )
            build_overwrite = gr.Checkbox(
                False,
                placeholder="overwrite",
                label="Overwrite Checkpoint if exist (else will add number)",
                elem_id="faceswaplab_build_overwrite",
            )
            generate_checkpoint_btn = gr.Button(
                "Save", elem_id="faceswaplab_build_save_btn"
            )
        with gr.Tab("Compare"):
            gr.Markdown(
                """Give a similarity score between two images (only first face is compared)."""
            )

            with gr.Row():
                img1 = gr.components.Image(
                    type="pil", label="Face 1", elem_id="faceswaplab_compare_face1"
                )
                img2 = gr.components.Image(
                    type="pil", label="Face 2", elem_id="faceswaplab_compare_face2"
                )
            compare_btn = gr.Button("Compare", elem_id="faceswaplab_compare_btn")
            compare_result_text = gr.Textbox(
                interactive=False,
                label="Similarity",
                value="0",
                elem_id="faceswaplab_compare_result",
            )
        with gr.Tab("Extract"):
            gr.Markdown(
                """Extract all faces from a batch of images. Will apply enhancement in the tools enhancement tab."""
            )
            with gr.Row():
                extracted_source_files = gr.components.File(
                    type="file",
                    file_count="multiple",
                    label="Batch Sources Images",
                    optional=True,
                    elem_id="faceswaplab_extract_batch_images",
                )
                extracted_faces = gr.Gallery(
                    label="Extracted faces",
                    show_label=False,
                    elem_id="faceswaplab_extract_results",
                )
            extract_save_path = gr.Textbox(
                label="Destination Directory",
                value="",
                elem_id="faceswaplab_extract_destination",
            )
            extract_btn = gr.Button("Extract", elem_id="faceswaplab_extract_btn")
        with gr.Tab("Explore Model"):
            model = gr.Dropdown(
                choices=models,
                label="Model not found, please download one and reload automatic 1111",
                elem_id="faceswaplab_explore_model",
            )
            explore_btn = gr.Button("Explore", elem_id="faceswaplab_explore_btn")
            explore_result_text = gr.Dataframe(
                interactive=False,
                label="Explored",
                elem_id="faceswaplab_explore_result",
            )
        with gr.Tab("Analyse Face"):
            img_to_analyse = gr.components.Image(
                type="pil", label="Face", elem_id="faceswaplab_analyse_face"
            )
            analyse_det_threshold = gr.Slider(
                0.1,
                1,
                0.5,
                step=0.01,
                label="Detection threshold",
                elem_id="faceswaplab_analyse_det_threshold",
            )
            analyse_btn = gr.Button("Analyse", elem_id="faceswaplab_analyse_btn")
            analyse_results = gr.Textbox(
                label="Results",
                interactive=False,
                value="",
                elem_id="faceswaplab_analyse_results",
            )

    with gr.Tab("Batch Process"):
        with gr.Tab("Source Images"):
            gr.Markdown(
                """Batch process images. Will apply enhancement in the tools enhancement tab."""
            )
            with gr.Row():
                batch_source_files = gr.components.File(
                    type="file",
                    file_count="multiple",
                    label="Batch Sources Images",
                    optional=True,
                    elem_id="faceswaplab_batch_images",
                )
                batch_results = gr.Gallery(
                    label="Batch result",
                    show_label=False,
                    elem_id="faceswaplab_batch_results",
                )
            batch_save_path = gr.Textbox(
                label="Destination Directory",
                value="outputs/faceswap/",
                elem_id="faceswaplab_batch_destination",
            )
            batch_save_btn = gr.Button(
                "Process & Save", elem_id="faceswaplab_extract_btn"
            )
        unit_components = []
        for i in range(1, get_sd_option("faceswaplab_units_count", 3) + 1):
            unit_components += faceswap_unit_ui(False, i, id_prefix="faceswaplab_tab")

    upscale_options = postprocessing_ui()

    explore_btn.click(
        explore_onnx_faceswap_model, inputs=[model], outputs=[explore_result_text]
    )
    compare_btn.click(compare, inputs=[img1, img2], outputs=[compare_result_text])
    generate_checkpoint_btn.click(
        build_face_checkpoint_and_save,
        inputs=[build_batch_files, build_name, build_gender, build_overwrite],
        outputs=[preview],
    )
    extract_btn.click(
        extract_faces,
        inputs=[extracted_source_files, extract_save_path] + upscale_options,
        outputs=[extracted_faces],
    )
    analyse_btn.click(
        analyse_faces,
        inputs=[img_to_analyse, analyse_det_threshold],
        outputs=[analyse_results],
    )
    batch_save_btn.click(
        batch_process,
        inputs=[batch_source_files, batch_save_path]
        + unit_components
        + upscale_options,
        outputs=[batch_results],
    )


def on_ui_tabs() -> List[Any]:
    with gr.Blocks(analytics_enabled=False) as ui_faceswap:
        tools_ui()
    return [(ui_faceswap, "FaceSwapLab", "faceswaplab_tab")]
