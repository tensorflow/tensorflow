# Build configurations for TensorRT.

def if_tensorrt(if_true, if_false=[]):
  """Tests whether TensorRT was enabled during the configure process."""
  return %{if_tensorrt}

def if_tensorrt_exec(if_true, if_false=[]):
  """Synonym for if_tensorrt."""
  return %{if_tensorrt}
