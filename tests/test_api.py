from typing import List
import pytest
import requests
import sys
import tempfile
import safetensors

sys.path.append(".")

import requests
from client_api.api_utils import (
    FaceSwapUnit,
    InswappperOptions,
    pil_to_base64,
    PostProcessingOptions,
    InpaintingWhen,
    InpaintingOptions,
    FaceSwapRequest,
    FaceSwapResponse,
    FaceSwapExtractRequest,
    FaceSwapCompareRequest,
    FaceSwapExtractResponse,
    compare_faces,
    base64_to_pil,
    base64_to_safetensors,
    safetensors_to_base64,
)
from PIL import Image

base_url = "http://127.0.0.1:7860"


@pytest.fixture
def face_swap_request() -> FaceSwapRequest:
    # First face unit
    unit1 = FaceSwapUnit(
        source_img=pil_to_base64("references/man.png"),  # The face you want to use
        faces_index=(0,),  # Replace first face
    )

    # Second face unit
    unit2 = FaceSwapUnit(
        source_img=pil_to_base64("references/woman.png"),  # The face you want to use
        same_gender=True,
        faces_index=(0,),  # Replace first woman since same gender is on
        swapping_options=InswappperOptions(
            face_restorer_name="CodeFormer",
            upscaler_name="LDSR",
            improved_mask=True,
            sharpen=True,
            color_corrections=True,
        ),
    )

    # Post-processing config
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
        image=pil_to_base64("tests/test_image.png"),
        units=[unit1, unit2],
        postprocessing=pp,
    )

    return request


def test_version() -> None:
    response = requests.get(f"{base_url}/faceswaplab/version")
    assert response.status_code == 200
    assert "version" in response.json()


def test_compare() -> None:
    request = FaceSwapCompareRequest(
        image1=pil_to_base64("references/man.png"),
        image2=pil_to_base64("references/man.png"),
    )

    response = requests.post(
        url=f"{base_url}/faceswaplab/compare",
        data=request.json(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200
    similarity = float(response.text)
    assert similarity > 0.90


def test_extract() -> None:
    pp = PostProcessingOptions(
        face_restorer_name="CodeFormer",
        codeformer_weight=0.5,
        restorer_visibility=1,
        upscaler_name="Lanczos",
    )

    request = FaceSwapExtractRequest(
        images=[pil_to_base64("tests/test_image.png")], postprocessing=pp
    )

    response = requests.post(
        url=f"{base_url}/faceswaplab/extract",
        data=request.json(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200

    res = FaceSwapExtractResponse.parse_obj(response.json())

    assert len(res.pil_images) == 2

    # First face is the man
    assert (
        compare_faces(
            res.pil_images[0], Image.open("tests/test_image.png"), base_url=base_url
        )
        > 0.5
    )


def test_faceswap(face_swap_request: FaceSwapRequest) -> None:
    response = requests.post(
        f"{base_url}/faceswaplab/swap_face",
        data=face_swap_request.json(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "images" in data
    assert "infos" in data

    res = FaceSwapResponse.parse_obj(response.json())
    images: List[Image.Image] = res.pil_images
    assert len(images) == 1
    image = images[0]
    orig_image = base64_to_pil(face_swap_request.image)
    assert image.width == orig_image.width * face_swap_request.postprocessing.scale
    assert image.height == orig_image.height * face_swap_request.postprocessing.scale

    # Compare the result and ensure similarity for the man (first face)

    request = FaceSwapCompareRequest(
        image1=pil_to_base64("references/man.png"),
        image2=res.images[0],
    )

    response = requests.post(
        url=f"{base_url}/faceswaplab/compare",
        data=request.json(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200
    similarity = float(response.text)
    assert similarity > 0.50


def test_faceswap_inpainting(face_swap_request: FaceSwapRequest) -> None:
    face_swap_request.units[0].pre_inpainting = InpaintingOptions(
        inpainting_denoising_strengh=0.4,
        inpainting_prompt="Photo of a funny man",
        inpainting_negative_prompt="blurry, bad art",
        inpainting_steps=100,
    )

    face_swap_request.units[0].post_inpainting = InpaintingOptions(
        inpainting_denoising_strengh=0.4,
        inpainting_prompt="Photo of a funny man",
        inpainting_negative_prompt="blurry, bad art",
        inpainting_steps=20,
        inpainting_sampler="Euler a",
    )

    response = requests.post(
        f"{base_url}/faceswaplab/swap_face",
        data=face_swap_request.json(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "images" in data
    assert "infos" in data


def test_faceswap_checkpoint_building() -> None:
    source_images: List[str] = [
        pil_to_base64("references/man.png"),
        pil_to_base64("references/woman.png"),
    ]

    response = requests.post(
        url=f"{base_url}/faceswaplab/build",
        json=source_images,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200

    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        base64_to_safetensors(response.json(), output_path=temp_file.name)
        with safetensors.safe_open(temp_file.name, framework="pt") as f:
            assert "age" in f.keys()
            assert "gender" in f.keys()
            assert "embedding" in f.keys()


def test_faceswap_checkpoint_building_and_using() -> None:
    source_images: List[str] = [
        pil_to_base64("references/man.png"),
    ]

    response = requests.post(
        url=f"{base_url}/faceswaplab/build",
        json=source_images,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200

    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        base64_to_safetensors(response.json(), output_path=temp_file.name)
        with safetensors.safe_open(temp_file.name, framework="pt") as f:
            assert "age" in f.keys()
            assert "gender" in f.keys()
            assert "embedding" in f.keys()

        # First face unit :
        unit1 = FaceSwapUnit(
            source_face=safetensors_to_base64(
                temp_file.name
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
        request = FaceSwapRequest(
            image=pil_to_base64("tests/test_image.png"), units=[unit1]
        )

        # Face Swap
        response = requests.post(
            url=f"{base_url}/faceswaplab/swap_face",
            data=request.json(),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        assert response.status_code == 200
        fsr = FaceSwapResponse.parse_obj(response.json())
        data = response.json()
        assert "images" in data
        assert "infos" in data

        # First face is the man
        assert (
            compare_faces(
                fsr.pil_images[0], Image.open("references/man.png"), base_url=base_url
            )
            > 0.5
        )
