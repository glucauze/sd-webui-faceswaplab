import launch
import os
import sys
import pkg_resources
from modules import shared
from packaging.version import parse


def check_install() -> None:
    use_gpu = getattr(
        shared.cmd_opts, "faceswaplab_gpu", False
    ) or shared.opts.data.get("faceswaplab_use_gpu", False)

    if use_gpu and sys.platform != "darwin":
        req_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "requirements-gpu.txt"
        )
    else:
        req_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "requirements.txt"
        )

    def is_installed(package: str) -> bool:
        package_name = package.split("==")[0].split(">=")[0].strip()
        try:
            installed_version = parse(
                pkg_resources.get_distribution(package_name).version
            )
        except pkg_resources.DistributionNotFound:
            return False

        if "==" in package:
            required_version = parse(package.split("==")[1])
            return installed_version == required_version
        elif ">=" in package:
            required_version = parse(package.split(">=")[1])
            return installed_version >= required_version
        else:
            return True

    print("Checking faceswaplab requirements")
    with open(req_file) as file:
        for package in file:
            try:
                package = package.strip()

                if not is_installed(package):
                    print(f"Install {package}")
                    launch.run_pip(
                        f"install {package}",
                        f"sd-webui-faceswaplab requirement: {package}",
                    )

            except Exception as e:
                print(e)
                print(
                    f"Warning: Failed to install {package}, faceswaplab will not work."
                )
                raise e


check_install()
