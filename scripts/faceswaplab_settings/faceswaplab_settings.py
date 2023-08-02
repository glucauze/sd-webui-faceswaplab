from scripts.faceswaplab_utils.models_utils import get_models
from modules import script_callbacks, shared
import gradio as gr


def on_ui_settings() -> None:
    section = ("faceswaplab", "FaceSwapLab")
    models = get_models()
    shared.opts.add_option(
        "faceswaplab_model",
        shared.OptionInfo(
            models[0] if len(models) > 0 else "None",
            "FaceSwapLab FaceSwap Model",
            gr.Dropdown,
            {"interactive": True, "choices": models},
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_keep_original",
        shared.OptionInfo(
            False,
            "keep original image before swapping",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_units_count",
        shared.OptionInfo(
            3,
            "Max faces units (requires restart)",
            gr.Slider,
            {"minimum": 1, "maximum": 10, "step": 1},
            section=section,
        ),
    )

    shared.opts.add_option(
        "faceswaplab_detection_threshold",
        shared.OptionInfo(
            0.5,
            "Face Detection threshold",
            gr.Slider,
            {"minimum": 0.1, "maximum": 0.99, "step": 0.001},
            section=section,
        ),
    )

    # DEFAULT UI SETTINGS

    shared.opts.add_option(
        "faceswaplab_pp_default_face_restorer",
        shared.OptionInfo(
            None,
            "UI Default global post processing face restorer (requires restart)",
            gr.Dropdown,
            {
                "interactive": True,
                "choices": ["None"] + [x.name() for x in shared.face_restorers],
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_pp_default_face_restorer_visibility",
        shared.OptionInfo(
            1,
            "UI Default global  post processing face restorer visibility (requires restart)",
            gr.Slider,
            {"minimum": 0, "maximum": 1, "step": 0.001},
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_pp_default_face_restorer_weight",
        shared.OptionInfo(
            1,
            "UI Default global post processing face restorer weight (requires restart)",
            gr.Slider,
            {"minimum": 0, "maximum": 1, "step": 0.001},
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_pp_default_upscaler",
        shared.OptionInfo(
            None,
            "UI Default global post processing upscaler (requires restart)",
            gr.Dropdown,
            {
                "interactive": True,
                "choices": [upscaler.name for upscaler in shared.sd_upscalers],
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_pp_default_upscaler_visibility",
        shared.OptionInfo(
            1,
            "UI Default global post processing upscaler visibility(requires restart)",
            gr.Slider,
            {"minimum": 0, "maximum": 1, "step": 0.001},
            section=section,
        ),
    )

    # Inpainting

    shared.opts.add_option(
        "faceswaplab_pp_default_inpainting_prompt",
        shared.OptionInfo(
            "Portrait of a [gender]",
            "UI Default inpainting prompt [gender] is replaced by man or woman (requires restart)",
            gr.Textbox,
            {},
            section=section,
        ),
    )

    shared.opts.add_option(
        "faceswaplab_pp_default_inpainting_negative_prompt",
        shared.OptionInfo(
            "blurry",
            "UI Default inpainting negative prompt [gender] (requires restart)",
            gr.Textbox,
            {},
            section=section,
        ),
    )

    # UPSCALED SWAPPER

    shared.opts.add_option(
        "faceswaplab_default_upscaled_swapper_upscaler",
        shared.OptionInfo(
            None,
            "Default Upscaled swapper upscaler (Recommanded : LDSR but slow) (requires restart)",
            gr.Dropdown,
            {
                "interactive": True,
                "choices": [upscaler.name for upscaler in shared.sd_upscalers],
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_default_upscaled_swapper_sharpen",
        shared.OptionInfo(
            False,
            "Default Upscaled swapper sharpen",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_default_upscaled_swapper_fixcolor",
        shared.OptionInfo(
            False,
            "Default Upscaled swapper color corrections (requires restart)",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_default_upscaled_swapper_improved_mask",
        shared.OptionInfo(
            True,
            "Default Use improved segmented mask (use pastenet to mask only the face) (requires restart)",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_default_upscaled_swapper_face_restorer",
        shared.OptionInfo(
            None,
            "Default Upscaled swapper face restorer (requires restart)",
            gr.Dropdown,
            {
                "interactive": True,
                "choices": ["None"] + [x.name() for x in shared.face_restorers],
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_default_upscaled_swapper_face_restorer_visibility",
        shared.OptionInfo(
            1,
            "Default Upscaled swapper face restorer visibility (requires restart)",
            gr.Slider,
            {"minimum": 0, "maximum": 1, "step": 0.001},
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_default_upscaled_swapper_face_restorer_weight",
        shared.OptionInfo(
            1,
            "Default Upscaled swapper face restorer weight (codeformer) (requires restart)",
            gr.Slider,
            {"minimum": 0, "maximum": 1, "step": 0.001},
            section=section,
        ),
    )
    shared.opts.add_option(
        "faceswaplab_default_upscaled_swapper_erosion",
        shared.OptionInfo(
            1,
            "Default Upscaled swapper mask erosion factor, 1 = default behaviour. The larger it is, the more blur is applied around the face. Too large and the facial change is no longer visible. (requires restart)",
            gr.Slider,
            {"minimum": 0, "maximum": 10, "step": 0.001},
            section=section,
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)
