"""External versions of build rules that differ outside of Google."""

load(
    "//tensorflow:tensorflow.bzl",
    "clean_dep",
)

# Dependencies for the bzl_library rule for this file.
# This should include bzl_library targets for the bzl files loaded by the "load" statements above.
SPECIAL_RULES_DEPS = [
    "//tensorflow:tensorflow_bzl",
]

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

def ios_visibility_allowlist():
    """This is a no-op outside of Google."""
    pass

def internal_visibility_allowlist():
    """Grant public visibility to internal targets so that other repos can depend on them."""
    return ["//visibility:public"]

def jni_utils_visibility_allowlist():
    """Returns a list of packages that can depend on tensorflow/lite/java/src/main/native:jni_utils."""
    return ["//tensorflow/lite:__subpackages__"]

def nonportable_visibility_allowlist():
    """Grant public visibility to nonportable targets so that other repos can depend on them."""
    return ["//visibility:public"]

def op_resolver_internal_visibility_allowlist():
    """Returns a list of packages that can depend on tensorflow/lite/core/api:op_resolver_internal.

    This is a no-op outside of Google."""
    return []

def c_api_opaque_internal_visibility_allowlist():
    """Returns a list of packages that can depend on tensorflow/lite/c:c_api_opaque_internal.

    This is a no-op outside of Google."""
    return []

def nnapi_plugin_impl_visibility_allowlist():
    """Returns a list of packages that can depend on tensorflow/lite/acceleration/configuration:nnapi_plugin_impl.

    This is a no-op outside of Google."""
    return []

def nnapi_sl_visibility_allowlist():
    """Returns a list of packages that can depend on tensorflow/lite/nnapi/sl:nnapi_support_library_headers.

    This is a no-op outside of Google."""
    return []

def nnapi_native_srcs_visibility_allowlist():
    """Returns a list of packages that can depend on tensorflow/lite/delegates/nnapi/java/src/main/native:native_srcs

    This is a no-op outside of Google."""
    return []

def verifier_internal_visibility_allowlist():
    """Returns a list of packages that can depend on tensorflow/lite/tools:verifier_internal.

    This is a no-op outside of Google."""
    return []

def gpu_compatibility_without_gl_deps_internal_visibility_allowlist():
    """Returns a list of packages that can depend on tensorflow/lite/experimental/acceleration/compatibility:gpu_compatibility_without_gl_deps.

    This is a no-op outside of Google."""
    return []

def xnnpack_plugin_impl_visibility_allowlist():
    """Returns a list of packages that can depend on tensorflow/lite/core/acceleration/configuration:xnnpack_plugin.

    This is a no-op outside of Google."""
    return []

def tflite_internal_cc_3p_api_deps_src_all_visibility_allowlist():
    """Returns a list of packages that can depend on tensorflow/lite:tflite_internal_cc_3p_api_deps_src_all.

    This is a no-op outside of Google."""
    return []

def tflite_extra_gles_deps():
    """This is a no-op outside of Google."""
    return []

def tflite_ios_lab_runner(version):
    """This is a no-op outside of Google."""

    # Can switch back to None when https://github.com/bazelbuild/rules_apple/pull/757 is fixed
    return "@build_bazel_rules_apple//apple/testing/default_runner:ios_default_runner"

def if_nnapi(supported, not_supported = [], supported_android = None):
    if supported_android == None:
        supported_android = supported

    # We use a denylist rather than a allowlist for known unsupported platforms.
    return select({
        clean_dep("//tensorflow:emscripten"): not_supported,
        clean_dep("//tensorflow:ios"): not_supported,
        clean_dep("//tensorflow:macos"): not_supported,
        clean_dep("//tensorflow:windows"): not_supported,
        clean_dep("//tensorflow:android"): supported_android,
        "//conditions:default": supported,
    })

def tflite_hexagon_mobile_test(name):
    """This is a no-op outside of Google."""
    pass

def tflite_hexagon_nn_skel_libraries():
    """This is a no-op outside of Google due to license agreement process.

    Developers who want to use hexagon nn skel libraries can download
    and install the libraries as the guided in
    https://www.tensorflow.org/lite/performance/hexagon_delegate#step_2_add_hexagon_libraries_to_your_android_app.
    For example, if you installed the libraries at third_party/hexagon_nn_skel
    and created third_party/hexagon_nn_skel/BUILD with a build target,
    filegroup(
        name = "libhexagon_nn_skel",
        srcs = glob(["*.so"]),
    )
    you need to modify this macro to specify the build target.
    return ["//third_party/hexagon_nn_skel:libhexagon_nn_skel"]
    """
    return []

def tflite_schema_utils_friends():
    """This is a no-op outside of Google.

    Return the package group declaration to which targets for Flatbuffer schema utilities."""

    # Its usage should be rare, and is often abused by tools that are doing
    # Flatbuffer creation/manipulation in unofficially supported ways."
    return ["//..."]

def flex_portable_tensorflow_deps():
    """Returns dependencies for building portable tensorflow in Flex delegate."""

    return [
        "//third_party/fft2d:fft2d_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@eigen_archive//:eigen3",
        "@gemmlowp",
        "@icu//:common",
        "//third_party/icu/data:conversion_data",
    ]

def tflite_copts_extra():
    """Defines extra compile time flags for tflite_copts(). Currently empty."""
    return []

def tflite_extra_arm_config_settings():
    """Defines extra ARM CPU config_setting targets. Currently empty."""
    return []
