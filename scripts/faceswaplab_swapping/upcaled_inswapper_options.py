from dataclasses import *
from client_api import api_utils


@dataclass
class InswappperOptions:
    face_restorer_name: str = None
    restorer_visibility: float = 1
    codeformer_weight: float = 1
    upscaler_name: str = None
    improved_mask: bool = False
    color_corrections: bool = False
    sharpen: bool = False
    erosion_factor: float = 1.0

    @staticmethod
    def from_api_dto(dto: api_utils.InswappperOptions) -> "InswappperOptions":
        """
        Converts a InpaintingOptions object from an API DTO (Data Transfer Object).

        :param options: An object of api_utils.InpaintingOptions representing the
                        post-processing options as received from the API.
        :return: A InpaintingOptions instance containing the translated values
                from the API DTO.
        """
        if dto is None:
            return InswappperOptions()

        return InswappperOptions(
            face_restorer_name=dto.face_restorer_name,
            restorer_visibility=dto.restorer_visibility,
            codeformer_weight=dto.codeformer_weight,
            upscaler_name=dto.upscaler_name,
            improved_mask=dto.improved_mask,
            color_corrections=dto.color_corrections,
            sharpen=dto.sharpen,
            erosion_factor=dto.erosion_factor,
        )
