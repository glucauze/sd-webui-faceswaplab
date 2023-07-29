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
from scripts.faceswaplab_postprocessing.postprocessing_options import InpaintingWhen


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


def get_postprocessing_options(
    options: api_utils.PostProcessingOptions,
) -> PostProcessingOptions:
    pp_options = PostProcessingOptions(
        face_restorer_name=options.face_restorer_name,
        restorer_visibility=options.restorer_visibility,
        codeformer_weight=options.codeformer_weight,
        upscaler_name=options.upscaler_name,
        scale=options.scale,
        upscale_visibility=options.upscaler_visibility,
        inpainting_denoising_strengh=options.inpainting_denoising_strengh,
        inpainting_prompt=options.inpainting_prompt,
        inpainting_negative_prompt=options.inpainting_negative_prompt,
        inpainting_steps=options.inpainting_steps,
        inpainting_sampler=options.inpainting_sampler,
        # hacky way to prevent having a separate file for Inpainting when (2 classes)
        # therfore a conversion is required from api IW to server side IW
        inpainting_when=InpaintingWhen(options.inpainting_when.value),
        inpainting_model=options.inpainting_model,
    )

    assert isinstance(
        pp_options.inpainting_when, InpaintingWhen
    ), "Value is not a valid InpaintingWhen enum"

    return pp_options


def get_faceswap_units_settings(
    api_units: List[api_utils.FaceSwapUnit],
) -> List[FaceSwapUnitSettings]:
    units = []
    for u in api_units:
        units.append(
            FaceSwapUnitSettings(
                source_img=base64_to_pil(u.source_img),
                source_face=u.source_face,
                _batch_files=u.get_batch_images(),
                blend_faces=u.blend_faces,
                enable=True,
                same_gender=u.same_gender,
                sort_by_size=u.sort_by_size,
                check_similarity=u.check_similarity,
                _compute_similarity=u.compute_similarity,
                min_ref_sim=u.min_ref_sim,
                min_sim=u.min_sim,
                _faces_index=",".join([str(i) for i in (u.faces_index)]),
                reference_face_index=u.reference_face_index,
                swap_in_generated=True,
                swap_in_source=False,
            )
        )
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
                pp_options = get_postprocessing_options(request.postprocessing)
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
