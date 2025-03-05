"""Open source build configurations for CUDA."""

# This file is used in OSS only. It is not transformed by copybara. Therefore all paths in this
# file are OSS paths.

load(
    "@local_config_cuda//cuda:build_defs.bzl",
    _if_cuda_is_configured = "if_cuda_is_configured",
    _if_cuda_newer_than = "if_cuda_newer_than",
)

# IMPORTANT: Do not remove this load statement. We rely on that //xla/tsl doesn't exist in g3
# to prevent g3 .bzl files from loading this file.
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

# We perform this indirection so that the copybara tool can distinguish this
# macro from others provided by the same file.
def if_cuda_is_configured(x, no_cuda = []):
    return _if_cuda_is_configured(x, no_cuda)

# Constructs rpath linker flags for use with nvidia wheel-packaged libs
# avaialble from PyPI. Two paths are needed because symbols are used from
# both the root of the TensorFlow installation directory as well as from
# various pywrap libs within the 'python' subdir.
def cuda_rpath_flags(relpath):
    return [
        "-Wl,-rpath='$$ORIGIN/../../" + relpath + "'",
        "-Wl,-rpath='$$ORIGIN/../" + relpath + "'",
    ]

def if_cuda_newer_than(wanted_ver, if_true, if_false = []):
    return _if_cuda_newer_than(wanted_ver, if_true, if_false)
