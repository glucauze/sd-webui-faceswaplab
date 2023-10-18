from typing import List
import gradio as gr
from modules import sd_models, sd_samplers
from scripts.faceswaplab_utils.sd_utils import get_sd_option


def face_inpainting_ui(id_prefix: str = "faceswaplab") -> List[gr.components.Component]:
    inpainting_denoising_strength = gr.Slider(
        0,
        1,
        0,
        step=0.01,
        elem_id=f"{id_prefix}_pp_inpainting_denoising_strength",
        label="Denoising strength",
    )

    inpainting_denoising_prompt = gr.Textbox(
        get_sd_option(
            "faceswaplab_pp_default_inpainting_prompt", "Portrait of a [gender]"
        ),
        elem_id=f"{id_prefix}_pp_inpainting_denoising_prompt",
        label="Inpainting prompt use [gender] instead of men or woman",
    )
    inpainting_denoising_negative_prompt = gr.Textbox(
        get_sd_option("faceswaplab_pp_default_inpainting_negative_prompt", "blurry"),
        elem_id=f"{id_prefix}_pp_inpainting_denoising_neg_prompt",
        label="Inpainting negative prompt use [gender] instead of men or woman",
    )
    with gr.Row():
        samplers_names = [s.name for s in sd_samplers.all_samplers]
        inpainting_sampler = gr.Dropdown(
            choices=samplers_names,
            value=[samplers_names[0]],
            label="Inpainting Sampler",
            elem_id=f"{id_prefix}_pp_inpainting_sampler",
        )
        inpainting_denoising_steps = gr.Slider(
            1,
            150,
            20,
            step=1,
            label="Inpainting steps",
            elem_id=f"{id_prefix}_pp_inpainting_steps",
        )

    inpaiting_model = gr.Dropdown(
        choices=["Current"] + sd_models.checkpoint_tiles(),
        default="Current",
        label="sd model (experimental)",
        elem_id=f"{id_prefix}_pp_inpainting_sd_model",
    )

    inpaiting_seed = gr.Number(
        label="Inpainting seed",
        value=0,
        minimum=0,
        precision=0,
        elem_id=f"{id_prefix}_pp_inpainting_seed",
    )

    gradio_components: List[gr.components.Component] = [
        inpainting_denoising_strength,
        inpainting_denoising_prompt,
        inpainting_denoising_negative_prompt,
        inpainting_denoising_steps,
        inpainting_sampler,
        inpaiting_model,
        inpaiting_seed,
    ]

    for component in gradio_components:
        setattr(component, "do_not_save_to_config", True)

    return gradio_components
