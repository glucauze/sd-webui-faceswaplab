from typing import List
from scripts.faceswaplab_ui.faceswaplab_inpainting_ui import face_inpainting_ui
from scripts.faceswaplab_swapping.face_checkpoints import get_face_checkpoints
import gradio as gr
from modules import shared
from scripts.faceswaplab_utils.sd_utils import get_sd_option


def faceswap_unit_advanced_options(
    is_img2img: bool, unit_num: int = 1, id_prefix: str = "faceswaplab_"
) -> List[gr.components.Component]:
    with gr.Accordion(f"Post-Processing & Advanced Mask Options", open=False):
        gr.Markdown(
            """Post-processing and mask settings for unit faces. Best result : checks all, use LDSR, use Codeformer"""
        )
        with gr.Row():
            face_restorer_name = gr.Radio(
                label="Restore Face",
                choices=["None"] + [x.name() for x in shared.face_restorers],
                value=get_sd_option(
                    "faceswaplab_default_upscaled_swapper_face_restorer",
                    "None",
                ),
                type="value",
                elem_id=f"{id_prefix}_face{unit_num}_face_restorer",
            )
            with gr.Column():
                face_restorer_visibility = gr.Slider(
                    0,
                    1,
                    value=get_sd_option(
                        "faceswaplab_default_upscaled_swapper_face_restorer_visibility",
                        1.0,
                    ),
                    step=0.001,
                    label="Restore visibility",
                    elem_id=f"{id_prefix}_face{unit_num}_face_restorer_visibility",
                )
                codeformer_weight = gr.Slider(
                    0,
                    1,
                    value=get_sd_option(
                        "faceswaplab_default_upscaled_swapper_face_restorer_weight", 1.0
                    ),
                    step=0.001,
                    label="codeformer weight",
                    elem_id=f"{id_prefix}_face{unit_num}_face_restorer_weight",
                )
        upscaler_name = gr.Dropdown(
            choices=[upscaler.name for upscaler in shared.sd_upscalers],
            value=get_sd_option("faceswaplab_default_upscaled_swapper_upscaler", ""),
            label="Upscaler",
            elem_id=f"{id_prefix}_face{unit_num}_upscaler",
        )

        improved_mask = gr.Checkbox(
            get_sd_option("faceswaplab_default_upscaled_swapper_improved_mask", False),
            interactive=True,
            label="Use improved segmented mask (use pastenet to mask only the face)",
            elem_id=f"{id_prefix}_face{unit_num}_improved_mask",
        )
        color_corrections = gr.Checkbox(
            get_sd_option("faceswaplab_default_upscaled_swapper_fixcolor", False),
            interactive=True,
            label="Use color corrections",
            elem_id=f"{id_prefix}_face{unit_num}_color_corrections",
        )
        sharpen_face = gr.Checkbox(
            get_sd_option("faceswaplab_default_upscaled_swapper_sharpen", False),
            interactive=True,
            label="sharpen face",
            elem_id=f"{id_prefix}_face{unit_num}_sharpen_face",
        )
        erosion_factor = gr.Slider(
            0.0,
            10.0,
            get_sd_option("faceswaplab_default_upscaled_swapper_erosion", 1.0),
            step=0.01,
            label="Upscaled swapper mask erosion factor, 1 = default behaviour.",
            elem_id=f"{id_prefix}_face{unit_num}_erosion_factor",
        )

    components = [
        face_restorer_name,
        face_restorer_visibility,
        codeformer_weight,
        upscaler_name,
        improved_mask,
        color_corrections,
        sharpen_face,
        erosion_factor,
    ]

    for component in components:
        setattr(component, "do_not_save_to_config", True)

    return components


def faceswap_unit_ui(
    is_img2img: bool, unit_num: int = 1, id_prefix: str = "faceswaplab"
) -> List[gr.components.Component]:
    with gr.Tab(f"Face {unit_num}"):
        with gr.Column():
            gr.Markdown(
                """Reference is an image. First face will be extracted.
            First face of batches sources will be extracted and used as input (or blended if blend is activated)."""
            )
            with gr.Row():
                img = gr.components.Image(
                    type="pil",
                    label="Reference",
                    elem_id=f"{id_prefix}_face{unit_num}_reference_image",
                )
                batch_files = gr.components.File(
                    type="file",
                    file_count="multiple",
                    label="Batch Sources Images",
                    optional=True,
                    elem_id=f"{id_prefix}_face{unit_num}_batch_source_face_files",
                )
            gr.Markdown(
                """Face checkpoint built with the checkpoint builder in tools. Will overwrite reference image."""
            )
            with gr.Row():
                face = gr.Dropdown(
                    choices=get_face_checkpoints(),
                    label="Face Checkpoint (precedence over reference face)",
                    elem_id=f"{id_prefix}_face{unit_num}_face_checkpoint",
                )
                refresh = gr.Button(
                    value="↻",
                    variant="tool",
                    elem_id=f"{id_prefix}_face{unit_num}_refresh_checkpoints",
                )

                def refresh_fn(selected: str):
                    return gr.Dropdown.update(
                        value=selected, choices=get_face_checkpoints()
                    )

                refresh.click(fn=refresh_fn, inputs=face, outputs=face)

            with gr.Row():
                enable = gr.Checkbox(
                    False,
                    placeholder="enable",
                    label="Enable",
                    elem_id=f"{id_prefix}_face{unit_num}_enable",
                )
                blend_faces = gr.Checkbox(
                    True,
                    placeholder="Blend Faces",
                    label="Blend Faces ((Source|Checkpoint)+References = 1)",
                    elem_id=f"{id_prefix}_face{unit_num}_blend_faces",
                    interactive=True,
                )

            gr.Markdown(
                """Select the face to be swapped, you can sort by size or use the same gender as the desired face:"""
            )
            with gr.Row():
                same_gender = gr.Checkbox(
                    False,
                    placeholder="Same Gender",
                    label="Same Gender",
                    elem_id=f"{id_prefix}_face{unit_num}_same_gender",
                )
                sort_by_size = gr.Checkbox(
                    False,
                    placeholder="Sort by size",
                    label="Sort by size (larger>smaller)",
                    elem_id=f"{id_prefix}_face{unit_num}_sort_by_size",
                )
            target_faces_index = gr.Textbox(
                value=f"{unit_num-1}",
                placeholder="Which face to swap (comma separated), start from 0 (by gender if same_gender is enabled)",
                label="Target face : Comma separated face number(s)",
                elem_id=f"{id_prefix}_face{unit_num}_target_faces_index",
            )
            gr.Markdown(
                """The following will only affect reference face image (and is not affected by sort by size) :"""
            )
            reference_faces_index = gr.Number(
                value=0,
                precision=0,
                minimum=0,
                placeholder="Which face to get from reference image start from 0",
                label="Reference source face : start from 0",
                elem_id=f"{id_prefix}_face{unit_num}_reference_face_index",
            )
            gr.Markdown(
                """Configure swapping. Swapping can occure before img2img, after or both :""",
                visible=is_img2img,
            )
            swap_in_source = gr.Checkbox(
                False,
                placeholder="Swap face in source image",
                label="Swap in source image (blended face)",
                visible=is_img2img,
                elem_id=f"{id_prefix}_face{unit_num}_swap_in_source",
            )
            swap_in_generated = gr.Checkbox(
                True,
                placeholder="Swap face in generated image",
                label="Swap in generated image",
                visible=is_img2img,
                elem_id=f"{id_prefix}_face{unit_num}_swap_in_generated",
            )

        gr.Markdown(
            """                                        
## Advanced Options 

**Simple :** If you have bad results and don't want to fine-tune here, just enable Codeformer in "Global Post-Processing".
Otherwise, read the [doc](https://glucauze.github.io/sd-webui-faceswaplab/doc/) to understand following options.              

"""
        )

        with gr.Accordion("Similarity", open=False):
            gr.Markdown("""Discard images with low similarity or no faces :""")
            with gr.Row():
                check_similarity = gr.Checkbox(
                    False,
                    placeholder="discard",
                    label="Check similarity",
                    elem_id=f"{id_prefix}_face{unit_num}_check_similarity",
                )
                compute_similarity = gr.Checkbox(
                    False,
                    label="Compute similarity",
                    elem_id=f"{id_prefix}_face{unit_num}_compute_similarity",
                )
            min_sim = gr.Slider(
                0,
                1,
                0,
                step=0.01,
                label="Min similarity",
                elem_id=f"{id_prefix}_face{unit_num}_min_similarity",
            )
            min_ref_sim = gr.Slider(
                0,
                1,
                0,
                step=0.01,
                label="Min reference similarity",
                elem_id=f"{id_prefix}_face{unit_num}_min_ref_similarity",
            )

        with gr.Accordion(label="Pre-Inpainting (before swapping)", open=False):
            gr.Markdown("Pre-inpainting sends face to inpainting before swapping")
            pre_inpainting = face_inpainting_ui(
                id_prefix=f"{id_prefix}_face{unit_num}_preinpainting",
            )

        options = faceswap_unit_advanced_options(is_img2img, unit_num, id_prefix)

        with gr.Accordion(label="Post-Inpainting (After swapping)", open=False):
            gr.Markdown("Pre-inpainting sends face to inpainting before swapping")
            post_inpainting = face_inpainting_ui(
                id_prefix=f"{id_prefix}_face{unit_num}_postinpainting",
            )

    gradio_components: List[gr.components.Component] = (
        [
            img,
            face,
            batch_files,
            blend_faces,
            enable,
            same_gender,
            sort_by_size,
            check_similarity,
            compute_similarity,
            min_sim,
            min_ref_sim,
            target_faces_index,
            reference_faces_index,
            swap_in_source,
            swap_in_generated,
        ]
        + pre_inpainting
        + options
        + post_inpainting
    )

    # If changed, you need to change FaceSwapUnitSettings accordingly
    # ORDER of parameters is IMPORTANT. It should match the result of FaceSwapUnitSettings
    return gradio_components
