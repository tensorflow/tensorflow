"""External versions of build rules that differ outside of Google."""

def tflite_portable_test_suite(**kwargs):
    """This is a no-op outside of Google."""
    _ignore = [kwargs]
    pass

def tflite_portable_test_suite_combined(**kwargs):
    """This is a no-op outside of Google."""
    _ignore = [kwargs]
    pass

def tflite_ios_per_kernel_test(**kwargs):
    """This is a no-op outside of Google."""
    _ignore = [kwargs]
    pass

def ios_visibility_whitelist():
    """This is a no-op outside of Google."""
    pass

def tflite_extra_gles_deps():
    """This is a no-op outside of Google."""
    return []

def tflite_ios_lab_runner(version):
    """This is a no-op outside of Google."""
    return None
