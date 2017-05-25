def sycl_library_path(name):
    return "lib/lib{}.so".format(name)

def readlink_command():
    return "readlink"
