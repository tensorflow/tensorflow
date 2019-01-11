def stream_executor_friends():
    return ["//tensorflow/..."]

def tf_additional_cuda_platform_deps():
  return []

def tf_additional_cuda_driver_deps():
  return ["@local_config_cuda//cuda:cuda_driver"]

def tf_additional_cudnn_plugin_deps():
  return []
