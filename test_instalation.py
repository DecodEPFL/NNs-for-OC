import importlib


def check_package(package):
    try:
        importlib.import_module(package)
        print(f"{package} is installed correctly.")
        return 0
    except ImportError:
        print(f"Error: {package} is not installed.")
        return 1

packages = ["torch", "matplotlib", "numpy"]
flag = 0
for package in packages:
    out = check_package(package)
    flag += out
if flag != 0:
    print("[WARNING] At least one package needs to be reinstalled.")
