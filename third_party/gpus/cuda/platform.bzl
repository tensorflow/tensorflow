CUDA_VERSION = ""
CUDNN_VERSION = ""
PLATFORM = ""

def cuda_sdk_version():
  return CUDA_VERSION

def cudnn_sdk_version():
  return CUDNN_VERSION

def cuda_library_path(name, version = cuda_sdk_version()):
  if PLATFORM == "Darwin":
    return "lib/lib{}.{}.dylib".format(name, version)
  else:
    return "lib64/lib{}.so.{}".format(name, version)

def cuda_static_library_path(name):
  if PLATFORM == "Darwin":
    return "lib/lib{}_static.a".format(name)
  else:
    return "lib64/lib{}_static.a".format(name)

def cudnn_library_path(version = cudnn_sdk_version()):
  if PLATFORM == "Darwin":
    return "lib/libcudnn.{}.dylib".format(version)
  else:
    return "lib64/libcudnn.so.{}".format(version)

def cupti_library_path(version = cuda_sdk_version()):
  if PLATFORM == "Darwin":
    return "extras/CUPTI/lib/libcupti.{}.dylib".format(version)
  else:
    return "extras/CUPTI/lib64/libcupti.so.{}".format(version)

def readlink_command():
  if PLATFORM == "Darwin":
    return "greadlink"
  else:
    return "readlink"
