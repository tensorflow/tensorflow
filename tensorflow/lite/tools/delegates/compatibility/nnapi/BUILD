# BUILD rules for NNAPI delegate compatibility checking.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "nnapi_compatibility_lib",
    srcs = [
        "nnapi_compatibility_lib.cc",
    ],
    hdrs = [
        "nnapi_compatibility_lib.h",
    ],
    deps = [
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
    ],
)

cc_library(
    name = "nnapi_delegate_compatibility_checker",
    srcs = ["nnapi_delegate_compatibility_checker.cc"],
    hdrs = [
        "nnapi_delegate_compatibility_checker.h",
    ],
    copts = ["-DNNAPI_VERBOSE_VALIDATION"],
    deps = [
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:util",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/nnapi:nnapi_lib",
        "//tensorflow/lite/tools/delegates/compatibility/common:delegate_compatibility_checker_base",
        "//tensorflow/lite/tools/delegates/compatibility/common:delegate_compatibility_checker_util",
        "//tensorflow/lite/tools/delegates/compatibility/common:online_helper_delegate",
        "//tensorflow/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//tensorflow/lite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "nnapi_compatibility_lib_test",
    srcs = [
        "nnapi_compatibility_lib_test.cc",
    ],
    deps = [
        ":nnapi_compatibility_lib",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_compatibility_checker_test",
    srcs = [
        "nnapi_delegate_compatibility_checker_test.cc",
    ],
    data = [
        "//tensorflow/lite:testdata/add.bin",
    ],
    deps = [
        ":nnapi_delegate_compatibility_checker",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
