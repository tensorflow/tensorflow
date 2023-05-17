load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "simple_delegate",
    srcs = [
        "simple_delegate.cc",
    ],
    hdrs = [
        "simple_delegate.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates:utils",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "simple_delegate_test",
    srcs = ["simple_delegate_test.cc"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/delegates/utils/dummy_delegate",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "simple_opaque_delegate",
    srcs = ["simple_opaque_delegate.cc"],
    hdrs = ["simple_opaque_delegate.h"],
    generate_opaque_delegate_target = True,
    tflite_deps = [
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
    ],
    deps = [
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:util",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

filegroup(
    name = "c_api_test_builtin_op_models",
    testonly = 1,
    srcs = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/conv_huge_im2col.bin",
        "//tensorflow/lite:testdata/multi_add.bin",
    ],
)

cc_test(
    name = "simple_opaque_delegate_test",
    srcs = ["simple_opaque_delegate_test.cc"],
    data = [":c_api_test_builtin_op_models"],
    deps = [
        ":simple_opaque_delegate",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates:delegate_test_util",
        "//tensorflow/lite/delegates/utils/experimental/sample_stable_delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "async_type_helpers",
    srcs = ["async_type_helpers.cc"],
    hdrs = ["async_type_helpers.h"],
    tflite_deps = [
        ":ret_macros",
        "//tensorflow/lite/async/interop/c:attribute_map",
        "//tensorflow/lite/async/interop/c:constants",
        "//tensorflow/lite/async/interop/c:types",
    ],
)

cc_library_with_tflite(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    tflite_deps = [
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/c:c_api_types",
        ":ret_macros",
    ],
    deps = [
        "//tensorflow/lite:minimal_logging",
        "@com_google_absl//absl/status",
    ],
)

cc_library_with_tflite(
    name = "ret_macros",
    srcs = [],
    hdrs = ["ret_macros.h"],
    tflite_deps = ["//tensorflow/lite/c:c_api_types"],
    deps = [
        "//tensorflow/lite:minimal_logging",
    ],
)

cc_library_with_tflite(
    name = "sync_fence",
    srcs = ["sync_fence.cc"],
    hdrs = ["sync_fence.h"],
    tflite_deps = [
        ":ret_macros",
    ],
    deps = [
        "//tensorflow/lite:minimal_logging",
        "@com_google_absl//absl/types:span",
    ],
)
