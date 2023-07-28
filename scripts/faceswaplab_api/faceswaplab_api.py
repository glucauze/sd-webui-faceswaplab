from PIL import Image
import numpy as np
from fastapi import FastAPI
from modules.api import api
from scripts.faceswaplab_api.faceswaplab_api_types import (
    FaceSwapRequest,
    FaceSwapResponse,
)
from scripts.faceswaplab_globals import VERSION_FLAG
import gradio as gr
from typing import Dict, List, Optional, Union
from scripts.faceswaplab_swapping import swapper
from scripts.faceswaplab_ui.faceswaplab_unit_settings import FaceSwapUnitSettings
from scripts.faceswaplab_utils.imgutils import (
    base64_to_pil,
)
from scripts.faceswaplab_utils.models_utils import get_current_model
from modules.shared import opts


def encode_to_base64(image: Union[str, Image.Image, np.ndarray]) -> str:
    """
    Encode an image to a base64 string.

    The image can be a file path (str), a PIL Image, or a NumPy array.

    Args:
        image (Union[str, Image.Image, np.ndarray]): The image to encode.

    Returns:
        str: The base64-encoded image if successful, otherwise an empty string.
    """
    if isinstance(image, str):
        return image
    elif isinstance(image, Image.Image):
        return api.encode_pil_to_base64(image)
    elif isinstance(image, np.ndarray):
        return encode_np_to_base64(image)
    else:
        return ""


def encode_np_to_base64(image: np.ndarray) -> str:
    """
    Encode a NumPy array to a base64 string.

    The array is first converted to a PIL Image, then encoded.

    Args:
        image (np.ndarray): The NumPy array to encode.

    Returns:
        str: The base64-encoded image.
    """
    pil = Image.fromarray(image)
    return api.encode_pil_to_base64(pil)


def faceswaplab_api(_: gr.Blocks, app: FastAPI) -> None:
    @app.get(
        "/faceswaplab/version",
        tags=["faceswaplab"],
        description="Get faceswaplab version",
    )
    async def version() -> Dict[str, str]:
        return {"version": VERSION_FLAG}

    # use post as we consider the method non idempotent (which is debatable)
    @app.post(
        "/faceswaplab/swap_face",
        tags=["faceswaplab"],
        description="Swap a face in an image using units",
    )
    async def swap_face(request: FaceSwapRequest) -> FaceSwapResponse:
        units: List[FaceSwapUnitSettings] = []
        src_image: Optional[Image.Image] = base64_to_pil(request.image)
        response = FaceSwapResponse(images=[], infos=[])
        if src_image is not None:
            for u in request.units:
                units.append(
                    FaceSwapUnitSettings(
                        source_img=base64_to_pil(u.source_img),
                        source_face=u.source_face,
                        _batch_files=u.get_batch_images(),
                        blend_faces=u.blend_faces,
                        enable=True,
                        same_gender=u.same_gender,
                        check_similarity=u.check_similarity,
                        _compute_similarity=u.compute_similarity,
                        min_ref_sim=u.min_ref_sim,
                        min_sim=u.min_sim,
                        _faces_index=",".join([str(i) for i in (u.faces_index)]),
                        swap_in_generated=True,
                        swap_in_source=False,
                    )
                )

            swapped_images = swapper.process_images_units(
                get_current_model(),
                images=[(src_image, None)],
                units=units,
                upscaled_swapper=opts.data.get("faceswaplab_upscaled_swapper", False),
            )
            for img, info in swapped_images:
                response.images.append(encode_to_base64(img))
                response.infos.append(info)

        return response
