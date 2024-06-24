load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_cc_library_with_c_headers_test")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# Client of this library is responsible to provide an implementation of
# for headers in //third_party/tensorflow/lite/nnapi:nnapi_implementation_headers
cc_library(
    name = "nnapi_delegate_no_nnapi_implementation",
    srcs = select({
        "//tensorflow:ios": [
            "nnapi_delegate_disabled.cc",
        ],
        "//tensorflow:windows": [
            "nnapi_delegate_disabled.cc",
        ],
        "//conditions:default": [
            "nnapi_delegate.cc",
            "nnapi_delegate_c_api.cc",
            "quant_lstm_sup.cc",
            "quant_lstm_sup.h",
        ],
    }),
    hdrs = [
        "nnapi_delegate.h",
        "nnapi_delegate_c_api.h",
        "nnapi_delegate_kernel.h",
        "nnapi_delegate_plugin.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates:serialization",
        "//tensorflow/lite/delegates:utils",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "//tensorflow/lite/nnapi:nnapi_implementation_headers",
        "//tensorflow/lite/nnapi:nnapi_lib",
        "//tensorflow/lite/nnapi:nnapi_util",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@farmhash_archive//:farmhash",
    ],
)

cc_library(
    name = "nnapi_delegate",
    hdrs = [
        "nnapi_delegate.h",
        "nnapi_delegate_c_api.h",
        "nnapi_delegate_kernel.h",
        "nnapi_delegate_plugin.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":nnapi_delegate_no_nnapi_implementation",
        "//tensorflow/lite:allocation",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates:serialization",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/nnapi:nnapi_lib",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "nnapi_delegate_c_api",
    hdrs = ["nnapi_delegate_c_api.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        ":nnapi_delegate",
        "//tensorflow/lite/core/c:common",
    ],
)

exports_files(["nnapi_delegate_c_api.h"])

cc_library(
    name = "nnapi_delegate_verbose_validation",
    srcs = select({
        "//tensorflow:ios": [
            "nnapi_delegate_disabled.cc",
        ],
        "//tensorflow:windows": [
            "nnapi_delegate_disabled.cc",
        ],
        "//conditions:default": [
            "nnapi_delegate.cc",
            "quant_lstm_sup.cc",
            "quant_lstm_sup.h",
        ],
    }),
    hdrs = [
        "nnapi_delegate.h",
        "nnapi_delegate_kernel.h",
        "nnapi_delegate_plugin.h",
    ],
    copts = ["-DNNAPI_VERBOSE_VALIDATION"],
    deps = [
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates:serialization",
        "//tensorflow/lite/delegates:utils",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/nnapi:nnapi_lib",
        "//tensorflow/lite/nnapi:nnapi_util",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@farmhash_archive//:farmhash",
    ],
)

cc_library(
    name = "acceleration_test_util",
    testonly = 1,
    srcs = [
        "acceleration_test_list.cc",
        "acceleration_test_util.cc",
    ],
    hdrs = ["acceleration_test_util.h"],
    deps = [
        ":nnapi_delegate",
        "//tensorflow/lite/kernels:acceleration_test_util_internal",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "nnapi_delegate_mock_test",
    testonly = 1,
    hdrs = [
        "nnapi_delegate_mock_test.h",
    ],
    copts = select({
        "//tensorflow:ios": [
            "-DNNAPI_DELEGATE_DISABLED",
        ],
        "//tensorflow:windows": [
            "-DNNAPI_DELEGATE_DISABLED",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":nnapi_delegate",
        "//tensorflow/lite/nnapi:nnapi_handler",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/nnapi:nnapi_lib",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "nnapi_delegate_test",
    size = "small",
    srcs = [
        "nnapi_delegate_test.cc",
    ],
    tags = [
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:deprecated_backends",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/nnapi:nnapi_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_c_api_test",
    size = "small",
    srcs = [
        "nnapi_delegate_c_api_test.cc",
    ],
    tags = [
        "no_windows",
        "tflite_not_portable_ios",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:deprecated_backends",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/nnapi:nnapi_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_errno_test",
    size = "small",
    srcs = [
        "nnapi_delegate_errno_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/nnapi:nnapi_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_device_selection_test",
    size = "small",
    srcs = [
        "nnapi_delegate_device_selection_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/nnapi:nnapi_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_nnapi_failure_handling_test",
    size = "small",
    srcs = [
        "nnapi_delegate_nnapi_failure_handling_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/nnapi:nnapi_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_signed_quantization_test",
    size = "small",
    srcs = [
        "nnapi_delegate_signed_quantization_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/nnapi:nnapi_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "quant_lstm_sup_test",
    size = "small",
    srcs = [
        "quant_lstm_sup.cc",
        "quant_lstm_sup.h",
        "quant_lstm_sup_test.cc",
    ],
    deps = [
        ":nnapi_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
