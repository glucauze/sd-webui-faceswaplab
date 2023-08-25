from dataclasses import dataclass
from typing import List, Optional
import gradio as gr
from client_api import api_utils


@dataclass
class InpaintingOptions:
    inpainting_denoising_strengh: float = 0
    inpainting_prompt: str = ""
    inpainting_negative_prompt: str = ""
    inpainting_steps: int = 20
    inpainting_sampler: str = "Euler"
    inpainting_model: str = "Current"
    inpainting_seed: int = -1

    @staticmethod
    def from_gradio(components: List[gr.components.Component]) -> "InpaintingOptions":
        return InpaintingOptions(*components)  # type: ignore

    @staticmethod
    def from_api_dto(dto: Optional[api_utils.InpaintingOptions]) -> "InpaintingOptions":
        """
        Converts a InpaintingOptions object from an API DTO (Data Transfer Object).

        :param options: An object of api_utils.InpaintingOptions representing the
                        post-processing options as received from the API.
        :return: A InpaintingOptions instance containing the translated values
                from the API DTO.
        """
        if dto is None:
            # Return default values
            return InpaintingOptions()

        return InpaintingOptions(
            inpainting_denoising_strengh=dto.inpainting_denoising_strengh,
            inpainting_prompt=dto.inpainting_prompt,
            inpainting_negative_prompt=dto.inpainting_negative_prompt,
            inpainting_steps=dto.inpainting_steps,
            inpainting_sampler=dto.inpainting_sampler,
            inpainting_model=dto.inpainting_model,
            inpainting_seed=dto.inpainting_seed,
        )
