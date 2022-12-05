load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_self_contained_libs_test_suite")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "builtin_ops",
    srcs = ["register.cc"],
    hdrs = [
        "builtin_op_kernels.h",
        "register.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite/core/shims:__subpackages__",
        "//tensorflow/lite/kernels:__subpackages__",
        "//tensorflow/lite/tools/benchmark:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:tflite_with_xnnpack_optional",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_op_kernels",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

# For internal usage by shared libraries only.
cc_library(
    name = "builtin_ops_all_linked",
    srcs = ["register.cc"],
    hdrs = [
        "builtin_op_kernels.h",
        "register.h",
    ],
    copts = tflite_copts(),
    # Limit visibility to TFLite only.
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:tflite_with_xnnpack_optional",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_op_kernels",
        "//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")
