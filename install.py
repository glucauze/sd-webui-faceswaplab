import launch
import os
import sys
import pkg_resources
from packaging.version import parse


def check_install() -> None:
    use_gpu = True
    try:
        from modules import shared

        use_gpu = not getattr(shared.cmd_opts, "use-cpu", False)
    except:
        # On some platform previous lines may failed (modules.shared not initialized), just ignore and use GPU requirements
        pass

    if use_gpu and sys.platform != "darwin":
        print("Faceswaplab : Use GPU requirements")
        req_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "requirements-gpu.txt"
        )
    else:
        print("Faceswaplab : Use CPU requirements")
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
            if package_name == "opencv-python":
                return launch.is_installed(package_name) or launch.is_installed("cv2")
            return launch.is_installed(package_name)

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
                    f"Warning: Failed to install {package}, faceswaplab may not work. Try to restart server or install dependencies manually."
                )
                raise e


import timeit

check_time = timeit.timeit(check_install, number=1)
print(check_time)
