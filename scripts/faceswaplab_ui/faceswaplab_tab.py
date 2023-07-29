import os
import tempfile
from pprint import pformat, pprint

import dill as pickle
import gradio as gr
import modules.scripts as scripts
import onnx
import pandas as pd
from scripts.faceswaplab_ui.faceswaplab_unit_ui import faceswap_unit_ui
from scripts.faceswaplab_ui.faceswaplab_upscaler_ui import upscaler_ui
from insightface.app.common import Face
from modules import scripts
from PIL import Image
from modules.shared import opts

from scripts.faceswaplab_utils import imgutils
from scripts.faceswaplab_utils.imgutils import pil_to_cv2
from scripts.faceswaplab_utils.models_utils import get_models
from scripts.faceswaplab_utils.faceswaplab_logging import logger
import scripts.faceswaplab_swapping.swapper as swapper
from scripts.faceswaplab_postprocessing.postprocessing_options import (
    PostProcessingOptions,
)
from scripts.faceswaplab_postprocessing.postprocessing import enhance_image
from dataclasses import fields
from typing import Any, Dict, List, Optional
from scripts.faceswaplab_ui.faceswaplab_unit_settings import FaceSwapUnitSettings
import re


def compare(img1: Image.Image, img2: Image.Image) -> str:
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

    return "You need 2 images to compare"


def extract_faces(
    files: List[gr.File],
    extract_path: Optional[str],
    *components: List[gr.components.Component],
) -> Optional[List[str]]:
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

    try:
        postprocess_options = PostProcessingOptions(*components)  # type: ignore

        if not extract_path:
            extract_path = tempfile.mkdtemp()

        if files:
            images = []
            for file in files:
                img = Image.open(file.name)
                faces = swapper.get_faces(pil_to_cv2(img))

                if faces:
                    face_images = []
                    for face in faces:
                        bbox = face.bbox.astype(int)
                        x_min, y_min, x_max, y_max = bbox
                        face_image = img.crop((x_min, y_min, x_max, y_max))

                        if (
                            postprocess_options.face_restorer_name
                            or postprocess_options.restorer_visibility
                        ):
                            postprocess_options.scale = (
                                1 if face_image.width > 512 else 512 // face_image.width
                            )
                            face_image = enhance_image(face_image, postprocess_options)

                        path = tempfile.NamedTemporaryFile(
                            delete=False, suffix=".png", dir=extract_path
                        ).name
                        face_image.save(path)
                        face_images.append(path)

                    images += face_images

            return images
    except Exception as e:
        logger.info("Failed to extract : %s", e)
        import traceback

        traceback.print_exc()
    return None


def analyse_faces(image: Image.Image, det_threshold: float = 0.5) -> Optional[str]:
    """
    Function to analyze the faces in an image and provide a detailed report.

    Parameters
    ----------
    image : PIL.Image.Image
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

    return None


def sanitize_name(name: str) -> str:
    logger.debug(f"Sanitize name {name}")
    name = re.sub("[^A-Za-z0-9_. ]+", "", name)
    name = name.replace(" ", "_")
    logger.debug(f"Sanitized name {name[:255]}")
    return name[:255]


def build_face_checkpoint_and_save(
    batch_files: gr.File, name: str
) -> Optional[Image.Image]:
    """
    Builds a face checkpoint using the provided image files, performs face swapping,
    and saves the result to a file. If a blended face is successfully obtained and the face swapping
    process succeeds, the resulting image is returned. Otherwise, None is returned.

    Args:
        batch_files (list): List of image file paths used to create the face checkpoint.
        name (str): The name assigned to the face checkpoint.

    Returns:
        PIL.Image.Image or None: The resulting swapped face image if the process is successful; None otherwise.
    """

    try:
        name = sanitize_name(name)
        batch_files = batch_files or []
        logger.info("Build %s %s", name, [x.name for x in batch_files])
        faces = swapper.get_faces_from_img_files(batch_files)
        blended_face = swapper.blend_faces(faces)
        preview_path = os.path.join(
            scripts.basedir(), "extensions", "sd-webui-faceswaplab", "references"
        )

        faces_path = os.path.join(scripts.basedir(), "models", "faceswaplab", "faces")

        os.makedirs(faces_path, exist_ok=True)

        target_img = None
        if blended_face:
            if blended_face["gender"] == 0:
                target_img = Image.open(os.path.join(preview_path, "woman.png"))
            else:
                target_img = Image.open(os.path.join(preview_path, "man.png"))

            if name == "":
                name = "default_name"
            pprint(blended_face)
            result = swapper.swap_face(
                blended_face, blended_face, target_img, get_models()[0]
            )
            result_image = enhance_image(
                result.image,
                PostProcessingOptions(
                    face_restorer_name="CodeFormer", restorer_visibility=1
                ),
            )

            file_path = os.path.join(faces_path, f"{name}.pkl")
            file_number = 1
            while os.path.exists(file_path):
                file_path = os.path.join(faces_path, f"{name}_{file_number}.pkl")
                file_number += 1
            result_image.save(file_path + ".png")
            with open(file_path, "wb") as file:
                pickle.dump(
                    {
                        "embedding": blended_face.embedding,
                        "gender": blended_face.gender,
                        "age": blended_face.age,
                    },
                    file,
                )
            try:
                with open(file_path, "rb") as file:
                    data = Face(pickle.load(file))
                    print(data)
            except Exception as e:
                print(e)
            return result_image

        print("No face found")
    except Exception as e:
        logger.error("Failed to build checkpoint %s", e)
        return None

    return target_img


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
        logger.info("Failed to explore model %s", e)
        return None
    return df


def batch_process(
    files: List[gr.File], save_path: str, *components: List[gr.components.Component]
) -> Optional[List[Image.Image]]:
    try:
        units_count = opts.data.get("faceswaplab_units_count", 3)
        units: List[FaceSwapUnitSettings] = []

        # Parse and convert units flat components into FaceSwapUnitSettings
        for i in range(0, units_count):
            units += [FaceSwapUnitSettings.get_unit_configuration(i, components)]  # type: ignore

        for i, u in enumerate(units):
            logger.debug("%s, %s", pformat(i), pformat(u))

        # Parse the postprocessing options
        # We must first find where to start from (after face swapping units)
        len_conf: int = len(fields(FaceSwapUnitSettings))
        shift: int = units_count * len_conf
        postprocess_options = PostProcessingOptions(
            *components[shift : shift + len(fields(PostProcessingOptions))]  # type: ignore
        )
        logger.debug("%s", pformat(postprocess_options))
        images = [
            Image.open(file.name) for file in files
        ]  # potentially greedy but Image.open is supposed to be lazy
        return swapper.batch_process(
            images,
            save_path=save_path,
            units=units,
            postprocess_options=postprocess_options,
        )
    except Exception as e:
        logger.error("Batch Process error : %s", e)
        import traceback

        traceback.print_exc()
    return None


def tools_ui() -> None:
    models = get_models()
    with gr.Tab("Tools"):
        with gr.Tab("Build"):
            gr.Markdown(
                """Build a face based on a batch list of images. Will blend the resulting face and store the checkpoint in the faceswaplab/faces directory."""
            )
            with gr.Row():
                batch_files = gr.components.File(
                    type="file",
                    file_count="multiple",
                    label="Batch Sources Images",
                    optional=True,
                    elem_id="faceswaplab_build_batch_files",
                )
                preview = gr.components.Image(
                    type="pil",
                    label="Preview",
                    interactive=False,
                    elem_id="faceswaplab_build_preview_face",
                )
            name = gr.Textbox(
                value="Face",
                placeholder="Name of the character",
                label="Name of the character",
                elem_id="faceswaplab_build_character_name",
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
                ).style(columns=[2], rows=[2])
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
                ).style(columns=[2], rows=[2])
            batch_save_path = gr.Textbox(
                label="Destination Directory",
                value="outputs/faceswap/",
                elem_id="faceswaplab_batch_destination",
            )
            batch_save_btn = gr.Button(
                "Process & Save", elem_id="faceswaplab_extract_btn"
            )
        unit_components = []
        for i in range(1, opts.data.get("faceswaplab_units_count", 3) + 1):
            unit_components += faceswap_unit_ui(False, i, id_prefix="faceswaplab_tab")

    upscale_options = upscaler_ui()

    explore_btn.click(
        explore_onnx_faceswap_model, inputs=[model], outputs=[explore_result_text]
    )
    compare_btn.click(compare, inputs=[img1, img2], outputs=[compare_result_text])
    generate_checkpoint_btn.click(
        build_face_checkpoint_and_save, inputs=[batch_files, name], outputs=[preview]
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
