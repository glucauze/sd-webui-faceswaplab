import launch
import os
import pkg_resources
import sys
from modules import shared

use_gpu = getattr(shared.cmd_opts, "faceswaplab_gpu", False)

if use_gpu and sys.platform != "darwin":
    req_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "requirements-gpu.txt"
    )
else:
    req_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "requirements.txt"
    )

print("Checking faceswaplab requirements")
with open(req_file) as file:
    for package in file:
        try:
            python = sys.executable
            package = package.strip()

            if not launch.is_installed(package.split("==")[0]):
                print(f"Install {package}")
                launch.run_pip(
                    f"install {package}", f"sd-webui-faceswaplab requirement: {package}"
                )
            elif "==" in package:
                package_name, package_version = package.split("==")
                installed_version = pkg_resources.get_distribution(package_name).version
                if installed_version != package_version:
                    print(
                        f"Install {package}, {installed_version} vs {package_version}"
                    )
                    launch.run_pip(
                        f"install {package}",
                        f"sd-webui-faceswaplab requirement: changing {package_name} version from {installed_version} to {package_version}",
                    )

        except Exception as e:
            print(e)
            print(f"Warning: Failed to install {package}, faceswaplab will not work.")
            raise e
