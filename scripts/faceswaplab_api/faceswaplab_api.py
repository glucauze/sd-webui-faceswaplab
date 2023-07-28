from PIL import Image
import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from modules.api.models import *
from modules.api import api
from scripts.faceswaplab_api.faceswaplab_api_types import (
    FaceSwapUnit,
    FaceSwapRequest,
    FaceSwapResponse,
)
from scripts.faceswaplab_globals import VERSION_FLAG
import gradio as gr
from typing import List, Optional
from scripts.faceswaplab_swapping import swapper
from scripts.faceswaplab_utils.faceswaplab_logging import save_img_debug
from scripts.faceswaplab_ui.faceswaplab_unit_settings import FaceSwapUnitSettings
from scripts.faceswaplab_utils.imgutils import (
    pil_to_cv2,
    check_against_nsfw,
    base64_to_pil,
)
from scripts.faceswaplab_utils.models_utils import get_current_model
from modules.shared import opts


def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return api.encode_pil_to_base64(image)
    elif type(image) is np.ndarray:
        return encode_np_to_base64(image)
    else:
        return ""


def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return api.encode_pil_to_base64(pil)


def faceswaplab_api(_: gr.Blocks, app: FastAPI):
    @app.get(
        "/faceswaplab/version",
        tags=["faceswaplab"],
        description="Get faceswaplab version",
    )
    async def version():
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
