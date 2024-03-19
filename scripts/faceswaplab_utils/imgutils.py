import io
from typing import List, Optional, Union, Dict
from PIL import Image
import cv2
import numpy as np
from math import isqrt, ceil
import torch
from modules import processing
import base64
from collections import Counter
from scripts.faceswaplab_utils.sd_utils import get_sd_option
from scripts.faceswaplab_utils.typing import BoxCoords, CV2ImgU8, PILImage
from scripts.faceswaplab_utils.faceswaplab_logging import logger


def check_against_nsfw(img: PILImage) -> bool:
    """
    Check if an image exceeds the Not Safe for Work (NSFW) score.

    Parameters:
    img (PILImage): The image to be checked.

    Returns:
    bool: True if any part of the image is considered NSFW, False otherwise.
    """

    NSFW_SCORE_THRESHOLD = get_sd_option("faceswaplab_nsfw_threshold", 0.7)

    # For testing purpose :
    if NSFW_SCORE_THRESHOLD >= 1:
        return False

    from ifnude import detect

    shapes: List[bool] = []
    chunks: List[Dict[str, Union[int, float]]] = detect(img)

    for chunk in chunks:
        logger.debug(
            f"chunck score {chunk['score']}, threshold : {NSFW_SCORE_THRESHOLD}"
        )
        shapes.append(chunk["score"] > NSFW_SCORE_THRESHOLD)

    return any(shapes)


def pil_to_cv2(pil_img: PILImage) -> CV2ImgU8:
    """
    Convert a PIL Image into an OpenCV image (cv2).

    Args:
        pil_img (PILImage): An image in PIL format.

    Returns:
        CV2ImgU8: The input image converted to OpenCV format (BGR).
    """
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR).astype("uint8")


def cv2_to_pil(cv2_img: CV2ImgU8) -> PILImage:  # type: ignore
    """
    Convert an OpenCV image (cv2) into a PIL Image.

    Args:
        cv2_img (CV2ImgU8): An image in OpenCV format (BGR).

    Returns:
        PILImage: The input image converted to PIL format (RGB).
    """
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def torch_to_pil(tensor: torch.Tensor) -> List[PILImage]:
    """
    Converts a tensor image or a batch of tensor images to a PIL image or a list of PIL images.

    Parameters
    ----------
    images : torch.Tensor
        A tensor representing an image or a batch of images.

    Returns
    -------
    list
        A list of PIL images.
    """
    images: CV2ImgU8 = tensor.cpu().permute(0, 2, 3, 1).numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def pil_to_torch(pil_images: Union[PILImage, List[PILImage]]) -> torch.Tensor:
    """
    Converts a PIL image or a list of PIL images to a torch tensor or a batch of torch tensors.

    Parameters
    ----------
    pil_images : Union[PILImage, List[PILImage]]
        A PIL image or a list of PIL images.

    Returns
    -------
    torch.Tensor
        A tensor representing an image or a batch of images.
    """
    if isinstance(pil_images, list):
        numpy_images = [np.array(image) for image in pil_images]
        torch_images = torch.from_numpy(np.stack(numpy_images)).permute(0, 3, 1, 2)
        return torch_images

    numpy_image = np.array(pil_images)
    torch_image = torch.from_numpy(numpy_image).permute(2, 0, 1)
    return torch_image


def create_square_image(image_list: List[PILImage]) -> Optional[PILImage]:
    """
    Creates a square image by combining multiple images in a grid pattern.

    Args:
        image_list (list): List of PIL Image objects to be combined.

    Returns:
        PIL Image object: The resulting square image.
        None: If the image_list is empty or contains only one image.
    """

    # Count the occurrences of each image size in the image_list
    size_counter = Counter(image.size for image in image_list)

    # Get the most common image size (size with the highest count)
    common_size = size_counter.most_common(1)[0][0]

    # Filter the image_list to include only images with the common size
    image_list = [image for image in image_list if image.size == common_size]

    # Get the dimensions (width and height) of the common size
    size = common_size

    # If there are more than one image in the image_list
    if len(image_list) > 1:
        num_images = len(image_list)

        # Calculate the number of rows and columns for the grid
        rows = isqrt(num_images)
        cols = ceil(num_images / rows)

        # Calculate the size of the square image
        square_size = (cols * size[0], rows * size[1])

        # Create a new RGB image with the square size
        square_image = Image.new("RGB", square_size)

        # Paste each image onto the square image at the appropriate position
        for i, image in enumerate(image_list):
            row = i // cols
            col = i % cols

            square_image.paste(image, (col * size[0], row * size[1]))

        # Return the resulting square image
        return square_image

    # Return None if there are no images or only one image in the image_list
    return None


def create_mask(
    image: PILImage,
    box_coords: BoxCoords,
) -> PILImage:
    """
    Create a binary mask for a given image and bounding box coordinates.

    Args:
        image (PILImage): The input image.
        box_coords (Tuple[int, int, int, int]): A tuple of 4 integers defining the bounding box.
        It follows the pattern (x1, y1, x2, y2), where (x1, y1) is the top-left coordinate of the
        box and (x2, y2) is the bottom-right coordinate of the box.

    Returns:
        PILImage: A binary mask of the same size as the input image, where pixels within
        the bounding box are white (255) and pixels outside the bounding box are black (0).
    """
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    x1, y1, x2, y2 = box_coords
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            mask.putpixel((x, y), 255)
    return mask


def apply_mask(
    img: PILImage, p: processing.StableDiffusionProcessing, batch_index: int
) -> PILImage:
    """
    Apply mask overlay and color correction to an image if enabled

    Args:
        img: PIL Image objects.
        p : The processing object
        batch_index : the batch index

    Returns:
        PIL Image object
    """
    if isinstance(p, processing.StableDiffusionProcessingImg2Img):
        if p.inpaint_full_res:
            overlays = p.overlay_images
            if overlays is None or batch_index >= len(overlays):
                return img
            overlay: PILImage = overlays[batch_index]
            logger.debug("Overlay size %s, Image size %s", overlay.size, img.size)
            if overlay.size != img.size:
                overlay = overlay.resize((img.size), resample=Image.Resampling.LANCZOS)
            img = img.copy()
            img.paste(overlay, (0, 0), overlay)
            return img

        if p.overlay_images and batch_index < len(p.overlay_images):
            selected_overlay = p.overlay_images[batch_index]
            img, _ = processing.apply_overlay(img, p.paste_to, selected_overlay)
        else:
            logger.debug("No overlay selected or overlay list is empty.")
            return img
        
        if p.color_corrections is not None and batch_index < len(p.color_corrections):
            img = processing.apply_color_correction(
                p.color_corrections[batch_index], img
            )
    return img


def prepare_mask(mask: PILImage, p: processing.StableDiffusionProcessing) -> PILImage:
    """
    Prepare an image mask for the inpainting process. (This comes from controlnet)

    This function takes as input a PIL Image object and an instance of the
    StableDiffusionProcessing class, and performs the following steps to prepare the mask:

    1. Convert the mask to grayscale (mode "L").
    2. If the 'inpainting_mask_invert' attribute of the processing instance is True,
       invert the mask colors.
    3. If the 'mask_blur' attribute of the processing instance is greater than 0,
       apply a Gaussian blur to the mask with a radius equal to 'mask_blur'.

    Args:
        mask (PILImage): The input mask as a PIL Image object.
        p (processing.StableDiffusionProcessing): An instance of the StableDiffusionProcessing class
                                                   containing the processing parameters.

    Returns:
        mask (PILImage): The prepared mask as a PIL Image object.
    """
    mask = mask.convert("L")
    # FIXME : Properly fix blur
    # if getattr(p, "mask_blur", 0) > 0:
    #     mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))
    return mask


def base64_to_pil(base64str: Optional[str]) -> Optional[PILImage]:
    """
    Converts a base64 string to a PIL Image object.

    Parameters:
    base64str (Optional[str]): The base64 string to convert. This string may contain a data URL scheme
    (i.e., 'data:image/jpeg;base64,') or just be the raw base64 encoded data. If None, the function
    will return None.

    Returns:
    Optional[PILImage]: A PIL Image object created from the base64 string. If the input is None,
    the function returns None.

    Raises:
    binascii.Error: If the base64 string is not properly formatted or encoded.
    PIL.UnidentifiedImageError: If the image format cannot be identified.
    """

    if base64str is None:
        return None

    # Check if the base64 string has a data URL scheme
    if "base64," in base64str:
        base64_data = base64str.split("base64,")[-1]
        img_bytes = base64.b64decode(base64_data)
    else:
        # If no data URL scheme, just decode
        img_bytes = base64.b64decode(base64str)

    return Image.open(io.BytesIO(img_bytes))
