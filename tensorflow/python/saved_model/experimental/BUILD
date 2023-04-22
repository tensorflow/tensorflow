# Copybara rules require the following `load` to be written exactly
# like that (in a single line, for example).
# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_pybind_cc_library_wrapper")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility =
        ["//tensorflow/python/saved_model:__subpackages__"],
    licenses = ["notice"],
)

tf_pybind_cc_library_wrapper(
    name = "pywrap_libexport_headers",
    deps = [
        "//tensorflow/cc/experimental/libexport:metrics",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_libexport",
    srcs = [
        "pywrap_libexport.cc",
        "pywrap_libexport_metrics.cc",
    ],
    hdrs = [
        "pywrap_libexport_metrics.h",
    ],
    features = ["-layering_check"],
    module_name = "pywrap_libexport",
    deps = [
        ":pywrap_libexport_headers",
        "//tensorflow/cc/experimental/libexport:constants",
        "//tensorflow/cc/experimental/libexport:save",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

tf_py_test(
    name = "pywrap_libexport_metrics_test",
    srcs = ["pywrap_libexport_metrics_test.py"],
    deps = [
        ":pywrap_libexport",
        "//tensorflow/python/eager:test",
    ],
)
