from typing import List
import gradio as gr
import modules
from modules import shared, sd_models
from scripts.faceswaplab_postprocessing.postprocessing_options import InpaintingWhen
from scripts.faceswaplab_utils.sd_utils import get_sd_option
from scripts.faceswaplab_ui.faceswaplab_inpainting_ui import face_inpainting_ui


def postprocessing_ui() -> List[gr.components.Component]:
    with gr.Tab(f"Global Post-Processing"):
        gr.Markdown(
            """Upscaling is performed on the whole image and all faces (including not swapped). Upscaling happens before face restoration."""
        )
        with gr.Row():
            face_restorer_name = gr.Radio(
                label="Restore Face",
                choices=["None"] + [x.name() for x in shared.face_restorers],
                value=get_sd_option(
                    "faceswaplab_pp_default_face_restorer",
                    shared.face_restorers[0].name(),
                ),
                type="value",
                elem_id="faceswaplab_pp_face_restorer",
            )

            with gr.Column():
                face_restorer_visibility = gr.Slider(
                    0,
                    1,
                    value=get_sd_option(
                        "faceswaplab_pp_default_face_restorer_visibility", 1
                    ),
                    step=0.001,
                    label="Restore visibility",
                    elem_id="faceswaplab_pp_face_restorer_visibility",
                )
                codeformer_weight = gr.Slider(
                    0,
                    1,
                    value=get_sd_option(
                        "faceswaplab_pp_default_face_restorer_weight", 1
                    ),
                    step=0.001,
                    label="codeformer weight",
                    elem_id="faceswaplab_pp_face_restorer_weight",
                )
        upscaler_name = gr.Dropdown(
            choices=[upscaler.name for upscaler in shared.sd_upscalers],
            value=get_sd_option("faceswaplab_pp_default_upscaler", "None"),
            label="Upscaler",
            elem_id="faceswaplab_pp_upscaler",
        )
        upscaler_scale = gr.Slider(
            1,
            8,
            1,
            step=0.1,
            label="Upscaler scale",
            elem_id="faceswaplab_pp_upscaler_scale",
        )
        upscaler_visibility = gr.Slider(
            0,
            1,
            value=get_sd_option("faceswaplab_pp_default_upscaler_visibility", 1),
            step=0.1,
            label="Upscaler visibility (if scale = 1)",
            elem_id="faceswaplab_pp_upscaler_visibility",
        )

        with gr.Accordion(label="Global-Inpainting (all faces)", open=False):
            gr.Markdown(
                "Inpainting sends image to inpainting with a mask on face (once for each faces)."
            )
            inpainting_when = gr.Dropdown(
                elem_id="faceswaplab_pp_inpainting_when",
                choices=[e.value for e in InpaintingWhen.__members__.values()],
                value=[InpaintingWhen.BEFORE_RESTORE_FACE.value],
                label="Enable/When",
            )
            global_inpainting = face_inpainting_ui("faceswaplab_gpp")

    components = [
        face_restorer_name,
        face_restorer_visibility,
        codeformer_weight,
        upscaler_name,
        upscaler_scale,
        upscaler_visibility,
        inpainting_when,
    ] + global_inpainting

    # Ask sd to not store in ui-config.json
    for component in components:
        setattr(component, "do_not_save_to_config", True)
    return components
