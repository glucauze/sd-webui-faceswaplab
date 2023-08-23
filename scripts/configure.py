import os
from tqdm import tqdm
import urllib.request
from scripts.faceswaplab_utils.faceswaplab_logging import logger
from scripts.faceswaplab_globals import *
from packaging import version
import pkg_resources
from scripts.faceswaplab_utils.models_utils import check_model

ALREADY_DONE = False


def check_configuration() -> None:
    global ALREADY_DONE

    if ALREADY_DONE:
        return

    # This has been moved here due to pb with sdnext in install.py not doing what a1111 is doing.
    models_dir = MODELS_DIR
    faces_dir = FACES_DIR

    model_url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
    model_name = os.path.basename(model_url)
    model_path = os.path.join(models_dir, model_name)

    def download(url: str, path: str) -> None:
        request = urllib.request.urlopen(url)
        total = int(request.headers.get("Content-Length", 0))
        with tqdm(
            total=total,
            desc="Downloading inswapper model",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            urllib.request.urlretrieve(
                url,
                path,
                reporthook=lambda count, block_size, total_size: progress.update(
                    block_size
                ),
            )

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)

    if not os.path.exists(model_path):
        download(model_url, model_path)
        check_model()

    gradio_version = pkg_resources.get_distribution("gradio").version

    if version.parse(gradio_version) < version.parse("3.32.0"):
        logger.warning(
            "Errors may occur with gradio versions lower than 3.32.0. Your version : %s",
            gradio_version,
        )

    ALREADY_DONE = True
