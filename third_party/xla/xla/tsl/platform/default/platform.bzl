"""Platform specific paths for various libraries and utilities."""
CUDA_VERSION = ""

CUDNN_VERSION = ""

PLATFORM = ""

def cuda_sdk_version():
    return CUDA_VERSION

def cudnn_sdk_version():
    return CUDNN_VERSION

# buildifier: disable=function-docstring
def cuda_library_path(name, version = cuda_sdk_version()):
    if PLATFORM == "Darwin":
        if not version:
            return "lib/lib{}.dylib".format(name)
        else:
            return "lib/lib{}.{}.dylib".format(name, version)
    elif not version:
        return "lib64/lib{}.so".format(name)
    else:
        return "lib64/lib{}.so.{}".format(name, version)

def cuda_static_library_path(name):
    if PLATFORM == "Darwin":
        return "lib/lib{}_static.a".format(name)
    else:
        return "lib64/lib{}_static.a".format(name)

# buildifier: disable=function-docstring
def cudnn_library_path(version = cudnn_sdk_version()):
    if PLATFORM == "Darwin":
        if not version:
            return "lib/libcudnn.dylib"
        else:
            return "lib/libcudnn.{}.dylib".format(version)
    elif not version:
        return "lib64/libcudnn.so"
    else:
        return "lib64/libcudnn.so.{}".format(version)

# buildifier: disable=function-docstring
def cupti_library_path(version = cuda_sdk_version()):
    if PLATFORM == "Darwin":
        if not version:
            return "extras/CUPTI/lib/libcupti.dylib"
        else:
            return "extras/CUPTI/lib/libcupti.{}.dylib".format(version)
    elif not version:
        return "extras/CUPTI/lib64/libcupti.so"
    else:
        return "extras/CUPTI/lib64/libcupti.so.{}".format(version)

def readlink_command():
    if PLATFORM == "Darwin":
        return "greadlink"
    else:
        return "readlink"
