package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "delegate_compatibility_checker_base",
    srcs = ["delegate_compatibility_checker_base.cc"],
    hdrs = ["delegate_compatibility_checker_base.h"],
    deps = [
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//tensorflow/lite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
    ],
)
