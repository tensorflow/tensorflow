load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")

def stream_executor_friends():
    return ["//tensorflow/..."]

def tf_additional_cuda_platform_deps():
    return []

def tf_additional_cuda_driver_deps():
    return ["//tensorflow/tsl/cuda:cuda_stub"]

def tf_additional_cupti_deps():
    return ["//tensorflow/compiler/xla/stream_executor/cuda:cupti_stub"]

def tf_additional_cudnn_plugin_deps():
    return []

def tf_additional_cudnn_plugin_copts():
    # TODO(timshen): remove TF_ENABLE_CUDNN_FRONTEND once cudnn-frontend is imported.
    return ["-DNV_CUDNN_DISABLE_EXCEPTION", "-DTF_ENABLE_CUDNN_FRONTEND"]

# Returns whether any GPU backend is configuered.
def if_gpu_is_configured(x):
    return if_cuda_is_configured(x) + if_rocm_is_configured(x)

def if_cuda_or_rocm(x):
    return if_gpu_is_configured(x)

# nvlink is not available via the pip wheels, disable it since it will create
# unnecessary dependency
def tf_additional_gpu_compilation_copts():
    return ["-DTF_DISABLE_NVLINK_BY_DEFAULT"]
