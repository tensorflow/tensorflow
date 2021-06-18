load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

pybind_extension(
    name = "_pywrap_analyzer_wrapper",
    srcs = [
        "analyzer_wrapper.cc",
    ],
    link_in_framework = True,
    module_name = "_pywrap_analyzer_wrapper",
    deps = [
        ":model_analyzer",
        "//tensorflow/compiler/mlir/lite:flatbuffer_to_mlir",
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
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
    ],
)
