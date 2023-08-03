from typing import List
import requests
from api_utils import (
    FaceSwapUnit,
    InswappperOptions,
    base64_to_safetensors,
    pil_to_base64,
    PostProcessingOptions,
    InpaintingWhen,
    InpaintingOptions,
    FaceSwapRequest,
    FaceSwapResponse,
    FaceSwapExtractRequest,
    FaceSwapCompareRequest,
    FaceSwapExtractResponse,
    safetensors_to_base64,
)

address = "http://127.0.0.1:7860"

# This has been tested on Linux platforms. This might requires some minor adaptations for windows.


#############################
# FaceSwap

# First face unit :
unit1 = FaceSwapUnit(
    source_img=pil_to_base64("../references/man.png"),  # The face you want to use
    faces_index=(0,),  # Replace first face
)

# Second face unit :
unit2 = FaceSwapUnit(
    source_img=pil_to_base64("../references/woman.png"),  # The face you want to use
    same_gender=True,
    faces_index=(0,),  # Replace first woman since same gender is on
)

# Post-processing config :
pp = PostProcessingOptions(
    face_restorer_name="CodeFormer",
    codeformer_weight=0.5,
    restorer_visibility=1,
    upscaler_name="Lanczos",
    scale=4,
    inpainting_when=InpaintingWhen.BEFORE_RESTORE_FACE,
    inpainting_options=InpaintingOptions(
        inpainting_steps=30,
        inpainting_denoising_strengh=0.1,
    ),
)

# Prepare the request
request = FaceSwapRequest(
    image=pil_to_base64("test_image.png"), units=[unit1, unit2], postprocessing=pp
)

# Face Swap
result = requests.post(
    url=f"{address}/faceswaplab/swap_face",
    data=request.json(),
    headers={"Content-Type": "application/json; charset=utf-8"},
)
response = FaceSwapResponse.parse_obj(result.json())

for img in response.pil_images:
    img.show()

#############################
# Comparison

request = FaceSwapCompareRequest(
    image1=pil_to_base64("../references/man.png"),
    image2=pil_to_base64(response.pil_images[0]),
)

result = requests.post(
    url=f"{address}/faceswaplab/compare",
    data=request.json(),
    headers={"Content-Type": "application/json; charset=utf-8"},
)

print("similarity", result.text)

#############################
# Extraction

# Prepare the request
request = FaceSwapExtractRequest(
    images=[pil_to_base64(response.pil_images[0])], postprocessing=pp
)

result = requests.post(
    url=f"{address}/faceswaplab/extract",
    data=request.json(),
    headers={"Content-Type": "application/json; charset=utf-8"},
)
response = FaceSwapExtractResponse.parse_obj(result.json())

for img in response.pil_images:
    img.show()


#############################
# Build checkpoint

source_images: List[str] = [
    pil_to_base64("../references/man.png"),
    pil_to_base64("../references/woman.png"),
]

result = requests.post(
    url=f"{address}/faceswaplab/build",
    json=source_images,
    headers={"Content-Type": "application/json; charset=utf-8"},
)

base64_to_safetensors(result.json(), output_path="test.safetensors")

#############################
# FaceSwap with local safetensors

# First face unit :
unit1 = FaceSwapUnit(
    source_face=safetensors_to_base64(
        "test.safetensors"
    ),  # convert the checkpoint to base64
    faces_index=(0,),  # Replace first face
    swapping_options=InswappperOptions(
        face_restorer_name="CodeFormer",
        upscaler_name="LDSR",
        improved_mask=True,
        sharpen=True,
        color_corrections=True,
    ),
)

# Prepare the request
request = FaceSwapRequest(image=pil_to_base64("test_image.png"), units=[unit1])

# Face Swap
result = requests.post(
    url=f"{address}/faceswaplab/swap_face",
    data=request.json(),
    headers={"Content-Type": "application/json; charset=utf-8"},
)
response = FaceSwapResponse.parse_obj(result.json())

for img in response.pil_images:
    img.show()
