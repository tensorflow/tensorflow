load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "format_converter_extension",
    srcs = ["format_converter_extension.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "format_converter_extension",
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools/optimize/sparsity:format_converter",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

py_test(
    name = "format_converter_extension_test",
    srcs = ["format_converter_extension_test.py"],
    python_version = "PY3",
    deps = [
        ":format_converter_extension",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
    ],
)
