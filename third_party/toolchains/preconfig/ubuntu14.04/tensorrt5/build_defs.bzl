# Build configurations for TensorRT.

def if_tensorrt(if_true, if_false = []):
    """Tests whether TensorRT was enabled during the configure process."""
    if True:
        return if_true
    return if_false
