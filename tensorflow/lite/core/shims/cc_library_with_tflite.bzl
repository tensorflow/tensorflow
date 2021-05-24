"""Definitions for targets that use the TFLite shims."""

load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("@build_bazel_rules_android//android:rules.bzl", "android_library")

def alias_with_tflite(name, actual, **kwargs):
    """Defines an alias for a target that uses the TFLite shims.

    This rule 'alias_with_tflite' should be used instead of the native
    'alias' rule whenever the 'actual' target that is being aliased
    is defined using one of the *_with_tflite build macros.

    Args:
      name: determines the name used for the alias target.
      actual: the target that the alias target is aliased to.
      **kwargs: additional alias parameters.
    """
    native.alias(name, actual, **kwargs)

def android_library_with_tflite(
        name,
        deps = [],
        tflite_deps = [],
        exports = [],
        tflite_exports = [],
        **kwargs):
    """Defines an android_library that uses the TFLite shims.

    This is a hook to allow applying different build flags (etc.)
    for targets that use the TFLite shims.

    Note that this build rule doesn't itself add any dependencies on
    TF Lite; this macro should normally be used in conjunction with a
    direct or indirect 'tflite_deps' dependency on one of the "shim"
    library targets from //third_party/tensorflow/lite/core/shims:*.

    Args:
      name: as for android_library.
      deps: as for android_library.
      tflite_deps: dependencies on rules that are themselves defined using
        'cc_library_with_tflite' / 'android_library_with_tflite'.
      exports: same as for android_library.
      tflite_exports: exported dependencies that are themselves defined using
        'cc_library_with_tflite' / 'android_library_with_tflite'.
      **kwargs: Additional android_library parameters.
    """
    android_library(
        name = name,
        exports = exports + tflite_exports,
        deps = deps + tflite_deps,
        **kwargs
    )

def cc_library_with_tflite(
        name,
        srcs = [],
        tflite_jni_binaries = [],
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
      srcs: as for cc_library.
      tflite_jni_binaries: dependencies on shared libraries that are defined
        using 'jni_binary_with_tflite'.
      deps: as for cc_library.
      tflite_deps: dependencies on rules that are themselves defined using
        'cc_library_with_tflite'.
      **kwargs: Additional cc_library parameters.
    """
    native.cc_library(
        name = name,
        srcs = srcs + tflite_jni_binaries,
        deps = deps + tflite_deps,
        **kwargs
    )

def cc_test_with_tflite(
        name,
        deps = [],
        tflite_deps = [],
        **kwargs):
    """Defines a cc_test that uses the TFLite shims.

    This is a hook to allow applying different build flags (etc.)
    for targets that use the TFLite shims.

    Note that this build rule doesn't itself add any dependencies on
    TF Lite; this macro should normally be used in conjunction with a
    direct or indirect 'tflite_deps' dependency on one of the "shim"
    library targets from //third_party/tensorflow/lite/core/shims:*.

    Args:
      name: as for cc_test.
      deps: as for cc_test.
      tflite_deps: dependencies on rules that are themselves defined using
        'cc_library_with_tflite'.
      **kwargs: Additional cc_test parameters.
    """
    native.cc_test(
        name = name,
        deps = deps + tflite_deps,
        **kwargs
    )

def jni_binary_with_tflite(
        name,
        deps = [],
        tflite_deps = [],
        **kwargs):
    """Defines a tflite_jni_binary that uses the TFLite shims.

    This is a hook to allow applying different build flags (etc.)
    for targets that use the TFLite shims.

    Note that this build rule doesn't itself add any dependencies on
    TF Lite; this macro should normally be used in conjunction with a
    direct or indirect 'tflite_deps' dependency on one of the "shim"
    library targets from //third_party/tensorflow/lite/core/shims:*.

    Args:
      name: as for tflite_jni_binary.
      deps: as for tflite_jni_binary.
      tflite_deps: dependencies on rules that are themselves defined using
        'cc_library_with_tflite'.
      **kwargs: Additional tflite_jni_binary parameters.
    """
    tflite_jni_binary(
        name = name,
        deps = deps + tflite_deps,
        **kwargs
    )
