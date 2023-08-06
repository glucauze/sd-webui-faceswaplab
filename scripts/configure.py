import os
from tqdm import tqdm
import traceback
import urllib.request
from scripts.faceswaplab_utils.faceswaplab_logging import logger
from scripts.faceswaplab_globals import *
from packaging import version
import pkg_resources
import hashlib

ALREADY_DONE = False


def check_install() -> None:
    # Very ugly hack :( due to sdnext optimization not calling install.py every time if git log has not changed
    import importlib.util
    import sys
    import os

    current_dir = os.path.dirname(os.path.realpath(__file__))
    check_install_path = os.path.join(current_dir, "..", "install.py")
    spec = importlib.util.spec_from_file_location("check_install", check_install_path)
    check_install = importlib.util.module_from_spec(spec)
    sys.modules["check_install"] = check_install
    spec.loader.exec_module(check_install)
    check_install.check_install()  # type: ignore
    #### End of ugly hack :( !


def is_sha1_matching(file_path: str, expected_sha1: str) -> bool:
    sha1_hash = hashlib.sha1(usedforsecurity=False)
    try:
        with open(file_path, "rb") as file:
            for byte_block in iter(lambda: file.read(4096), b""):
                sha1_hash.update(byte_block)
            if sha1_hash.hexdigest() == expected_sha1:
                return True
            else:
                return False
    except Exception as e:
        logger.error(
            "Failed to check model hash, check the model is valid or has been downloaded adequately : %e",
            e,
        )
        traceback.print_exc()
        return False


def check_configuration() -> None:
    global ALREADY_DONE

    if ALREADY_DONE:
        return

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

    if not os.path.exists(model_path):
        download(model_url, model_path)

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

    ALREADY_DONE = True
