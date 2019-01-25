def stream_executor_friends():
    return ["//tensorflow/..."]

def tf_additional_cuda_platform_deps():
  return []

# Use dynamic loading, therefore should be empty.
def tf_additional_cuda_driver_deps():
  return []

def tf_additional_cudnn_plugin_deps():
  return []
