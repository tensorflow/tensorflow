load("//tensorflow:tensorflow.default.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

pybind_extension(
    name = "_pywrap_analyzer_wrapper",
    srcs = [
        "analyzer_wrapper.cc",
    ],
    common_lib_packages = [
        "litert/python",
        "third_party/tensorflow/lite/python",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_analyzer_wrapper.pyi",
    ],
    wrap_py_init = True,
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
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/public:release_version",
        "//tensorflow/lite:version",
        "//tensorflow/lite/core:model_builder",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)
