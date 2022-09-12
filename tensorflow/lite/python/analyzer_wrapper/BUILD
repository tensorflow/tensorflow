load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

# Refer to: go/tf-pybind.
tf_python_pybind_extension(
    name = "_pywrap_analyzer_wrapper",
    srcs = [
        "analyzer_wrapper.cc",
    ],
    deps = [
        ":model_analyzer",
        "@pybind11",
    ],
)

cc_library(
    name = "model_analyzer",
    srcs = ["model_analyzer.cc"],
    hdrs = ["model_analyzer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite:version",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "//tensorflow/lite/tools/delegates/compatibility/gpu:gpu_delegate_compatibility_checker",
        "//tensorflow/lite/tools/delegates/compatibility/nnapi:nnapi_delegate_compatibility_checker",
        "//tensorflow/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//tensorflow/lite/tools/versioning:gpu_compatibility",
        "@com_google_absl//absl/strings",
    ],
)
