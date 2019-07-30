"""External versions of build rules that differ outside of Google."""

def tflite_portable_test_suite(**kwargs):
    """This is a no-op outside of Google."""
    _ignore = [kwargs]
    pass

def ios_visibility_whitelist():
    """This is a no-op outside of Google."""
    pass

def tflite_extra_gles_deps():
    """This is a no-op outside of Google."""
    return []
