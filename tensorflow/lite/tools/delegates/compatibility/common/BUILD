package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "online_helper_delegate",
    srcs = ["online_helper_delegate.cc"],
    hdrs = ["online_helper_delegate.h"],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "delegate_compatibility_checker_base",
    srcs = ["delegate_compatibility_checker_base.cc"],
    hdrs = [
        "delegate_compatibility_checker_base.h",
    ],
    deps = [
        ":delegate_compatibility_checker_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//tensorflow/lite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "delegate_compatibility_checker_util",
    hdrs = [
        "delegate_compatibility_checker_util.h",
    ],
)
