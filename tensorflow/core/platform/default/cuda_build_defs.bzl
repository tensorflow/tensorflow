"""Open source build configurations for CUDA."""

load("@local_config_cuda//cuda:build_defs.bzl", _if_cuda_is_configured = "if_cuda_is_configured")

# We perform this indirection so that the copybara tool can distinguish this
# macro from others provided by the same file.
def if_cuda_is_configured(x):
    return _if_cuda_is_configured(x)
