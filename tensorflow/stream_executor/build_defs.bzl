load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")

def stream_executor_friends():
    return ["//tensorflow/..."]

def tf_additional_cuda_platform_deps():
    return []

def tf_additional_cuda_driver_deps():
    return [":cuda_stub"]

def tf_additional_cupti_deps():
    return ["//tensorflow/stream_executor/cuda:cupti_stub"]

def tf_additional_cudnn_plugin_deps():
    return []

# Returns whether any GPU backend is configuered.
def if_gpu_is_configured(x):
    return if_cuda_is_configured(x) + if_rocm_is_configured(x)

def if_cuda_or_rocm(x):
    return if_gpu_is_configured(x)
