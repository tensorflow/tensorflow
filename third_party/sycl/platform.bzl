SYCL_VERSION = ""
PLATFORM = ""

def sycl_sdk_version():
  return SYCL_VERSION

def sycl_library_path(name, version = sycl_sdk_version()):
  if not version:
    return "lib/lib{}.so".format(name)
  else:
    return "lib/lib{}.so.{}".format(name, version)

def sycl_static_library_path(name):
  return "lib/lib{}_static.a".format(name)

def readlink_command():
    return "readlink"
