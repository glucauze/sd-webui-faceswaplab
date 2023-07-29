import requests
from api_utils import (
    FaceSwapRequest,
    FaceSwapUnit,
    PostProcessingOptions,
    FaceSwapResponse,
    pil_to_base64,
    InpaintingWhen,
    FaceSwapCompareRequest,
)

address = "http://127.0.0.1:7860"

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
    inpainting_steps=30,
    inpainting_denoising_strengh=0.1,
    inpainting_when=InpaintingWhen.BEFORE_RESTORE_FACE,
)

# Prepare the request
request = FaceSwapRequest(
    image=pil_to_base64("test_image.png"), units=[unit1, unit2], postprocessing=pp
)


result = requests.post(
    url=f"{address}/faceswaplab/swap_face",
    data=request.json(),
    headers={"Content-Type": "application/json; charset=utf-8"},
)
response = FaceSwapResponse.parse_obj(result.json())

for img in response.pil_images:
    img.show()


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
