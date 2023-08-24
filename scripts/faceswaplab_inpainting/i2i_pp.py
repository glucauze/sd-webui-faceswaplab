from scripts.faceswaplab_inpainting.faceswaplab_inpainting import InpaintingOptions
from scripts.faceswaplab_utils.faceswaplab_logging import logger
from PIL import Image
from modules import shared
from scripts.faceswaplab_utils import imgutils
from modules import shared, processing
from modules.processing import StableDiffusionProcessingImg2Img
from modules import sd_models
import traceback
from scripts.faceswaplab_swapping import swapper
from scripts.faceswaplab_utils.typing import *
from typing import *


def img2img_diffusion(
    img: PILImage, options: InpaintingOptions, faces: Optional[List[Face]] = None
) -> Image.Image:
    if not options or options.inpainting_denoising_strengh == 0:
        logger.info("Discard inpainting denoising strength is 0 or no inpainting")
        return img

    try:
        logger.info(
            f"""Inpainting face
Sampler : {options.inpainting_sampler}
inpainting_denoising_strength : {options.inpainting_denoising_strengh}
inpainting_steps : {options.inpainting_steps}
"""
        )
        if not isinstance(options.inpainting_sampler, str):
            options.inpainting_sampler = "Euler"

        logger.info("send faces to image to image")
        img = img.copy()

        if not faces:
            faces = swapper.get_faces(imgutils.pil_to_cv2(img))

        if faces:
            for face in faces:
                bbox = face.bbox.astype(int)
                mask = imgutils.create_mask(img, bbox)
                prompt = options.inpainting_prompt.replace(
                    "[gender]", "man" if face["gender"] == 1 else "woman"
                )
                negative_prompt = options.inpainting_negative_prompt.replace(
                    "[gender]", "man" if face["gender"] == 1 else "woman"
                )
                logger.info("Denoising prompt : %s", prompt)
                logger.info(
                    "Denoising strenght : %s", options.inpainting_denoising_strengh
                )

                i2i_kwargs = {
                    "sampler_name": options.inpainting_sampler,
                    "do_not_save_samples": True,
                    "steps": options.inpainting_steps,
                    "width": img.width,
                    "inpainting_fill": 1,
                    "inpaint_full_res": True,
                    "height": img.height,
                    "mask": mask,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "denoising_strength": options.inpainting_denoising_strengh,
                    "seed": options.inpainting_seed,
                }
                # Remove the following as they are not always supported on all platform :
                # "override_settings": {
                #     "return_mask_composite": False,
                #     "save_images_before_face_restoration": False,
                #     "save_images_before_highres_fix": False,
                #     "save_images_before_color_correction": False,
                #     "save_mask": False,
                #     "save_mask_composite": False,
                #     "samples_save": False,
                # },

                current_model_checkpoint = shared.opts.sd_model_checkpoint
                if options.inpainting_model and options.inpainting_model != "Current":
                    # Change checkpoint
                    shared.opts.sd_model_checkpoint = options.inpainting_model
                    sd_models.select_checkpoint
                    sd_models.load_model()
                i2i_p = StableDiffusionProcessingImg2Img([img], **i2i_kwargs)
                i2i_processed = processing.process_images(i2i_p)
                if options.inpainting_model and options.inpainting_model != "Current":
                    # Restore checkpoint
                    shared.opts.sd_model_checkpoint = current_model_checkpoint
                    sd_models.select_checkpoint
                    sd_models.load_model()

                images = i2i_processed.images
                if len(images) > 0:
                    img = images[0]
        return img
    except Exception as e:
        logger.error("Failed to apply inpainting to face : %s", e)
        traceback.print_exc()
        raise e
