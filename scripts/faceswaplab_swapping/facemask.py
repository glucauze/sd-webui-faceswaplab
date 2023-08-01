import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from scripts.faceswaplab_swapping.parsing import init_parsing_model
from functools import lru_cache
from typing import Union, List
from torch import device as torch_device


@lru_cache
def get_parsing_model(device: torch_device) -> torch.nn.Module:
    """
    Returns an instance of the parsing model.
    The returned model is cached for faster subsequent access.

    Args:
        device: The torch device to use for computations.

    Returns:
        The parsing model.
    """
    return init_parsing_model(device=device)  # type: ignore


def convert_image_to_tensor(
    images: Union[np.ndarray, List[np.ndarray]],
    convert_bgr_to_rgb: bool = True,
    use_float32: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Converts an image or a list of images to PyTorch tensor.

    Args:
        images: An image or a list of images in numpy.ndarray format.
        convert_bgr_to_rgb: A boolean flag indicating if the conversion from BGR to RGB should be performed.
        use_float32: A boolean flag indicating if the tensor should be converted to float32.

    Returns:
        PyTorch tensor or a list of PyTorch tensors.
    """

    def _convert_single_image_to_tensor(
        image: np.ndarray, convert_bgr_to_rgb: bool, use_float32: bool
    ) -> torch.Tensor:
        if image.shape[2] == 3 and convert_bgr_to_rgb:
            if image.dtype == "float64":
                image = image.astype("float32")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
        if use_float32:
            image_tensor = image_tensor.float()
        return image_tensor

    if isinstance(images, list):
        return [
            _convert_single_image_to_tensor(image, convert_bgr_to_rgb, use_float32)
            for image in images
        ]
    else:
        return _convert_single_image_to_tensor(images, convert_bgr_to_rgb, use_float32)


def generate_face_mask(face_image: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Generates a face mask given a face image.

    Args:
        face_image: The face image in numpy.ndarray format.
        device: The torch device to use for computations.

    Returns:
        The face mask as a numpy.ndarray.
    """
    # Resize the face image for the model
    resized_face_image = cv2.resize(
        face_image, (512, 512), interpolation=cv2.INTER_LINEAR
    )

    # Preprocess the image
    face_input = convert_image_to_tensor(
        (resized_face_image.astype("float32") / 255.0),
        convert_bgr_to_rgb=True,
        use_float32=True,
    )
    normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    assert isinstance(face_input, torch.Tensor)
    face_input = torch.unsqueeze(face_input, 0).to(device)

    # Pass the image through the model
    with torch.no_grad():
        model_output = get_parsing_model(device)(face_input)[0]
    model_output = model_output.argmax(dim=1).squeeze().cpu().numpy()

    # Generate the mask from the model output
    parse_mask = np.zeros(model_output.shape)
    MASK_COLOR_MAP = [
        0,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        0,
        255,
        0,
        0,
        0,
    ]
    for idx, color in enumerate(MASK_COLOR_MAP):
        parse_mask[model_output == idx] = color

    # Resize the mask to match the original image
    face_mask = cv2.resize(parse_mask, (face_image.shape[1], face_image.shape[0]))

    return face_mask
