import os
from tqdm import tqdm
import urllib.request
from scripts.faceswaplab_utils.faceswaplab_logging import logger
from scripts.faceswaplab_swapping.swapper import is_sha1_matching
from scripts.faceswaplab_utils.models_utils import get_models
from scripts.faceswaplab_globals import *
from packaging import version
import pkg_resources

ALREADY_DONE = False


def check_configuration() -> None:
    global ALREADY_DONE

    if ALREADY_DONE:
        return

    logger.info(f"FaceSwapLab {VERSION_FLAG} Config :")

    # This has been moved here due to pb with sdnext in install.py not doing what a1111 is doing.
    models_dir = MODELS_DIR
    faces_dir = FACES_DIR

    model_url = "https://huggingface.co/henryruhs/roop/resolve/main/inswapper_128.onnx"
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

    if not is_sha1_matching(model_path, EXPECTED_INSWAPPER_SHA1):
        logger.error(
            "Suspicious sha1 for model %s, check the model is valid or has been downloaded adequately. Should be %s",
            model_path,
            EXPECTED_INSWAPPER_SHA1,
        )

    gradio_version = pkg_resources.get_distribution("gradio").version

    if version.parse(gradio_version) < version.parse("3.32.0"):
        logger.warning(
            "Errors may occur with gradio versions lower than 3.32.0. Your version : %s",
            gradio_version,
        )

    if not os.path.exists(model_path):
        download(model_url, model_path)

    def print_infos() -> None:
        logger.info("FaceSwapLab config :")
        logger.info("+ MODEL DIR : %s", models_dir)
        models = get_models()
        logger.info("+ MODELS: %s", models)
        logger.info("+ FACES DIR : %s", faces_dir)
        logger.info("+ ANALYZER DIR : %s", ANALYZER_DIR)

    print_infos()

    ALREADY_DONE = True
