from PIL import Image
import numpy as np
from fastapi import FastAPI
from modules.api import api
from client_api.api_utils import (
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
from scripts.faceswaplab_postprocessing.postprocessing_options import (
    PostProcessingOptions,
)
from client_api import api_utils


def encode_to_base64(image: Union[str, Image.Image, np.ndarray]) -> str:  # type: ignore
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


def encode_np_to_base64(image: np.ndarray) -> str:  # type: ignore
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


def get_faceswap_units_settings(
    api_units: List[api_utils.FaceSwapUnit],
) -> List[FaceSwapUnitSettings]:
    units = []
    for u in api_units:
        units.append(FaceSwapUnitSettings.from_api_dto(u))
    return units


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
    async def swap_face(
        request: api_utils.FaceSwapRequest,
    ) -> api_utils.FaceSwapResponse:
        units: List[FaceSwapUnitSettings] = []
        src_image: Optional[Image.Image] = base64_to_pil(request.image)
        response = FaceSwapResponse(images=[], infos=[])

        if src_image is not None:
            if request.postprocessing:
                pp_options = PostProcessingOptions.from_api_dto(request.postprocessing)
            else:
                pp_options = None
            units = get_faceswap_units_settings(request.units)

            swapped_images = swapper.batch_process(
                [src_image], None, units=units, postprocess_options=pp_options
            )

            for img in swapped_images:
                response.images.append(encode_to_base64(img))

            response.infos = []  # Not used atm
        return response

    @app.post(
        "/faceswaplab/compare",
        tags=["faceswaplab"],
        description="Compare first face of each images",
    )
    async def compare(
        request: api_utils.FaceSwapCompareRequest,
    ) -> float:
        return swapper.compare_faces(
            base64_to_pil(request.image1), base64_to_pil(request.image2)
        )

    @app.post(
        "/faceswaplab/extract",
        tags=["faceswaplab"],
        description="Extract faces of each images",
    )
    async def extract(
        request: api_utils.FaceSwapExtractRequest,
    ) -> api_utils.FaceSwapExtractResponse:
        pp_options = None
        if request.postprocessing:
            pp_options = PostProcessingOptions.from_api_dto(request.postprocessing)
        images = [base64_to_pil(img) for img in request.images]
        faces = swapper.extract_faces(
            images, extract_path=None, postprocess_options=pp_options
        )
        result_images = [encode_to_base64(img) for img in faces]
        response = api_utils.FaceSwapExtractResponse(images=result_images)
        return response
