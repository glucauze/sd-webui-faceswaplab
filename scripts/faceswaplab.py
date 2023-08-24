from scripts.configure import check_configuration
from scripts.faceswaplab_utils.sd_utils import get_sd_option

check_configuration()

import importlib
import traceback

from scripts import faceswaplab_globals
from scripts.faceswaplab_api import faceswaplab_api
from scripts.faceswaplab_postprocessing import upscaling
from scripts.faceswaplab_settings import faceswaplab_settings
from scripts.faceswaplab_swapping import swapper
from scripts.faceswaplab_ui import faceswaplab_tab, faceswaplab_unit_ui
from scripts.faceswaplab_utils import faceswaplab_logging, imgutils, models_utils
from scripts.faceswaplab_utils.models_utils import get_current_swap_model
from scripts.faceswaplab_utils.typing import *
from scripts.faceswaplab_utils.ui_utils import dataclasses_from_flat_list
from scripts.faceswaplab_utils.faceswaplab_logging import logger, save_img_debug

# Reload all the modules when using "apply and restart"
# This is mainly done for development purposes
import logging

if logger.getEffectiveLevel() <= logging.DEBUG:
    importlib.reload(swapper)
    importlib.reload(faceswaplab_logging)
    importlib.reload(faceswaplab_globals)
    importlib.reload(imgutils)
    importlib.reload(upscaling)
    importlib.reload(faceswaplab_settings)
    importlib.reload(models_utils)
    importlib.reload(faceswaplab_unit_ui)
    importlib.reload(faceswaplab_api)

import os
from pprint import pformat
from typing import Any, List, Optional, Tuple

import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks, scripts, shared
from modules.images import save_image
from modules.processing import (
    Processed,
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
)
from modules.shared import opts

from scripts.faceswaplab_globals import VERSION_FLAG
from scripts.faceswaplab_postprocessing.postprocessing import enhance_image
from scripts.faceswaplab_postprocessing.postprocessing_options import (
    PostProcessingOptions,
)
from scripts.faceswaplab_ui.faceswaplab_unit_settings import FaceSwapUnitSettings

EXTENSION_PATH = os.path.join("extensions", "sd-webui-faceswaplab")


# Register the tab, done here to prevent it from being added twice
script_callbacks.on_ui_tabs(faceswaplab_tab.on_ui_tabs)

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(faceswaplab_api.faceswaplab_api)
except:
    logger.error("Failed to register API")

    traceback.print_exc()


class FaceSwapScript(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    @property
    def units_count(self) -> int:
        return get_sd_option("faceswaplab_units_count", 3)

    @property
    def enabled(self) -> bool:
        """Return True if any unit is enabled and the state is not interupted"""
        return any([u.enable for u in self.units]) and not shared.state.interrupted

    @property
    def keep_original_images(self) -> bool:
        return get_sd_option("faceswaplab_keep_original", False)

    @property
    def swap_in_generated_units(self) -> List[FaceSwapUnitSettings]:
        return [u for u in self.units if u.swap_in_generated and u.enable]

    @property
    def swap_in_source_units(self) -> List[FaceSwapUnitSettings]:
        return [u for u in self.units if u.swap_in_source and u.enable]

    def title(self) -> str:
        return f"faceswaplab"

    def show(self, is_img2img: bool) -> bool:
        return scripts.AlwaysVisible  # type: ignore

    def ui(self, is_img2img: bool) -> List[gr.components.Component]:
        with gr.Accordion(f"FaceSwapLab {VERSION_FLAG}", open=False):
            components: List[gr.components.Component] = []
            for i in range(1, self.units_count + 1):
                components += faceswaplab_unit_ui.faceswap_unit_ui(is_img2img, i)
            post_processing = faceswaplab_tab.postprocessing_ui()
        # If the order is modified, the before_process should be changed accordingly.

        components = components + post_processing

        return components

    def read_config(
        self, p: StableDiffusionProcessing, *components: Tuple[Any, ...]
    ) -> None:
        for i, c in enumerate(components):
            logger.debug("%s>%s", i, pformat(c))

        # The order of processing for the components is important
        # The method first process faceswap units then postprocessing units
        classes: List[Any] = dataclasses_from_flat_list(
            [FaceSwapUnitSettings] * self.units_count + [PostProcessingOptions],
            components,
        )
        self.units: List[FaceSwapUnitSettings] = []
        self.units += [u for u in classes if isinstance(u, FaceSwapUnitSettings)]
        self.postprocess_options = classes[-1]

        for i, u in enumerate(self.units):
            logger.debug("%s, %s", pformat(i), pformat(u))

        logger.debug("%s", pformat(self.postprocess_options))

        if self.enabled:
            p.do_not_save_samples = not self.keep_original_images

    def process(
        self, p: StableDiffusionProcessing, *components: Tuple[Any, ...]
    ) -> None:
        try:
            self.read_config(p, *components)

            # If is instance of img2img, we check if face swapping in source is required.
            if isinstance(p, StableDiffusionProcessingImg2Img):
                if self.enabled and len(self.swap_in_source_units) > 0:
                    init_images: List[Tuple[Optional[PILImage], Optional[str]]] = [
                        (img, None) for img in p.init_images
                    ]
                    new_inits = swapper.process_images_units(
                        get_current_swap_model(),
                        self.swap_in_source_units,
                        images=init_images,
                        force_blend=True,
                    )
                    logger.info(f"processed init images: {len(init_images)}")
                    if new_inits is not None:
                        p.init_images = [img[0] for img in new_inits]
        except Exception as e:
            logger.info("Failed to process : %s", e)
            traceback.print_exc()

    def postprocess(
        self, p: StableDiffusionProcessing, processed: Processed, *args: List[Any]
    ) -> None:
        try:
            if self.enabled:
                # Get the original images without the grid
                orig_images: List[PILImage] = processed.images[
                    processed.index_of_first_image :
                ]
                orig_infotexts: List[str] = processed.infotexts[
                    processed.index_of_first_image :
                ]

                keep_original = self.keep_original_images

                # These are were images and infos of swapped images will be stored
                images = []
                infotexts = []
                if (len(self.swap_in_generated_units)) > 0:
                    for i, (img, info) in enumerate(zip(orig_images, orig_infotexts)):
                        batch_index = i % p.batch_size
                        swapped_images = swapper.process_images_units(
                            get_current_swap_model(),
                            self.swap_in_generated_units,
                            images=[(img, info)],
                        )
                        if swapped_images is None:
                            continue

                        logger.info(f"{len(swapped_images)} images swapped")
                        for swp_img, new_info in swapped_images:
                            img = swp_img  # Will only swap the last image in the batch in next units (FIXME : hard to fix properly but not really critical)

                            if swp_img is not None:
                                save_img_debug(swp_img, "Before apply mask")
                                swp_img = imgutils.apply_mask(swp_img, p, batch_index)
                                save_img_debug(swp_img, "After apply mask")

                                try:
                                    if self.postprocess_options is not None:
                                        swp_img = enhance_image(
                                            swp_img, self.postprocess_options
                                        )
                                except Exception as e:
                                    logger.error("Failed to upscale : %s", e)

                                logger.info("Add swp image to processed")
                                images.append(swp_img)
                                infotexts.append(new_info)
                                if p.outpath_samples and opts.samples_save:
                                    save_image(
                                        swp_img,
                                        p.outpath_samples,
                                        "",
                                        p.all_seeds[batch_index],  # type: ignore
                                        p.all_prompts[batch_index],  # type: ignore
                                        opts.samples_format,
                                        info=new_info,
                                        p=p,
                                        suffix="-swapped",
                                    )
                            else:
                                logger.error("swp image is None")
                else:
                    keep_original = True

                # Generate grid :
                if opts.return_grid and len(images) > 1:
                    grid = imgutils.create_square_image(images)
                    text = processed.infotexts[0]
                    infotexts.insert(0, text)
                    if opts.enable_pnginfo:
                        grid.info["parameters"] = text  # type: ignore
                    images.insert(0, grid)

                    if opts.grid_save:
                        save_image(
                            grid,
                            p.outpath_grids,
                            "swapped-grid",
                            p.all_seeds[0],  # type: ignore
                            p.all_prompts[0],  # type: ignore
                            opts.grid_format,
                            info=text,
                            short_filename=not opts.grid_extended_filename,
                            p=p,
                            grid=True,
                        )

                if keep_original:
                    # If we want to keep original images, we add all existing (including grid this time)
                    images += processed.images
                    infotexts += processed.infotexts

                processed.images = images
                processed.infotexts = infotexts
        except Exception as e:
            logger.error("Failed to swap face in postprocess method : %s", e)
            traceback.print_exc()
