from typing import Optional
from modules.face_restoration import FaceRestoration
from modules.upscaler import UpscalerData
from dataclasses import dataclass
from modules import shared
from enum import Enum
from scripts.faceswaplab_inpainting.faceswaplab_inpainting import InpaintingOptions
from client_api import api_utils


class InpaintingWhen(Enum):
    NEVER = "Never"
    BEFORE_UPSCALING = "Before Upscaling/all"
    BEFORE_RESTORE_FACE = "After Upscaling/Before Restore Face"
    AFTER_ALL = "After All"


@dataclass
class PostProcessingOptions:
    face_restorer_name: str = ""
    restorer_visibility: float = 0.5
    codeformer_weight: float = 1

    upscaler_name: str = ""
    scale: float = 1
    upscale_visibility: float = 0.5

    inpainting_when: InpaintingWhen = InpaintingWhen.BEFORE_UPSCALING

    # (Don't use optional for this or gradio parsing will fail) :
    inpainting_options: InpaintingOptions = None  # type: ignore

    @property
    def upscaler(self) -> Optional[UpscalerData]:
        for upscaler in shared.sd_upscalers:
            if upscaler.name == self.upscaler_name:
                return upscaler
        return None

    @property
    def face_restorer(self) -> Optional[FaceRestoration]:
        for face_restorer in shared.face_restorers:
            if face_restorer.name() == self.face_restorer_name:
                return face_restorer
        return None

    @staticmethod
    def from_api_dto(
        options: api_utils.PostProcessingOptions,
    ) -> "PostProcessingOptions":
        """
        Converts a PostProcessingOptions object from an API DTO (Data Transfer Object).

        :param options: An object of api_utils.PostProcessingOptions representing the
                        post-processing options as received from the API.
        :return: A PostProcessingOptions instance containing the translated values
                from the API DTO.
        """
        return PostProcessingOptions(
            face_restorer_name=options.face_restorer_name,
            restorer_visibility=options.restorer_visibility,
            codeformer_weight=options.codeformer_weight,
            upscaler_name=options.upscaler_name,
            scale=options.scale,
            upscale_visibility=options.upscaler_visibility,
            inpainting_when=InpaintingWhen(options.inpainting_when.value),
            inpainting_options=InpaintingOptions.from_api_dto(
                options.inpainting_options
            ),
        )
