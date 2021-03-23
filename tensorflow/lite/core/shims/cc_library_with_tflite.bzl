"""Definitions for cc_library targets that use the TFLite shims."""

def cc_library_with_tflite(
        name,
        deps = [],
        tflite_deps = [],
        **kwargs):
    """Defines a cc_library that uses the TFLite shims.

    This is a hook to allow applying different build flags (etc.)
    for targets that use the TFLite shims.

    Note that this build rule doesn't itself add any dependencies on
    TF Lite; this macro should normally be used in conjunction with a
    direct or indirect 'tflite_deps' dependency on one of the "shim"
    library targets from //tensorflow/lite/core/shims:*.

    Args:
      name: as for cc_library.
      deps: as for cc_library.
      tflite_deps: dependencies on rules that are themselves defined using
        'cc_library_with_tflite'.
      **kwargs: Additional cc_library parameters.
    """
    native.cc_library(
        name = name,
        deps = deps + tflite_deps,
        **kwargs
    )
