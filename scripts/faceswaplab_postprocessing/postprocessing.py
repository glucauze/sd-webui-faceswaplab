from scripts.faceswaplab_utils.faceswaplab_logging import logger
from PIL import Image
from scripts.faceswaplab_postprocessing.postprocessing_options import (
    PostProcessingOptions,
    InpaintingWhen,
)
from scripts.faceswaplab_inpainting.i2i_pp import img2img_diffusion
from scripts.faceswaplab_postprocessing.upscaling import upscale_img, restore_face
import traceback


def enhance_image(image: Image.Image, pp_options: PostProcessingOptions) -> Image.Image:
    result_image = image
    try:
        logger.debug("enhance_image, inpainting : %s", pp_options.inpainting_when)
        result_image = image

        if (
            pp_options.inpainting_when == InpaintingWhen.BEFORE_UPSCALING.value
            or pp_options.inpainting_when == InpaintingWhen.BEFORE_UPSCALING
        ):
            logger.debug("Inpaint before upscale")
            result_image = img2img_diffusion(
                img=result_image, options=pp_options.inpainting_options
            )
        result_image = upscale_img(result_image, pp_options)

        if (
            pp_options.inpainting_when == InpaintingWhen.BEFORE_RESTORE_FACE.value
            or pp_options.inpainting_when == InpaintingWhen.BEFORE_RESTORE_FACE
        ):
            logger.debug("Inpaint before restore")
            result_image = img2img_diffusion(
                result_image, pp_options.inpainting_options
            )

        result_image = restore_face(result_image, pp_options)

        if (
            pp_options.inpainting_when == InpaintingWhen.AFTER_ALL.value
            or pp_options.inpainting_when == InpaintingWhen.AFTER_ALL
        ):
            logger.debug("Inpaint after all")
            result_image = img2img_diffusion(
                result_image, pp_options.inpainting_options
            )

    except Exception as e:
        logger.error("Failed to post-process %s", e)
        traceback.print_exc()
    return result_image
