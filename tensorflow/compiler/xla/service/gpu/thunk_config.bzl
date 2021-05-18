"""Thunk configuration for GpuExecutable."""

def if_using_bef_thunks(if_true, if_false = []):
    """Shorthand for select()'ing depending on whether we're using BEF thunks.

    Always return if_false until gpu_executable can depend on //third_party/tf_runtime/...

    """
    return if_false  # Change to 'if_true' to use BEF thunks.
