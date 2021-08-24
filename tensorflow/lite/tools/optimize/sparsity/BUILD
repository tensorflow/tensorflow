load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

pybind_extension(
    name = "format_converter_wrapper_pybind11",
    srcs = ["format_converter_wrapper_pybind11.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "format_converter_wrapper_pybind11",
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

py_test(
    name = "format_converter_wrapper_pybind11_test",
    srcs = ["format_converter_wrapper_pybind11_test.py"],
    python_version = "PY3",
    deps = [
        ":format_converter_wrapper_pybind11",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
    ],
)
