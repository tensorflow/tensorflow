# Copybara rules require the following `load` to be written exactly
# like that (in a single line, for example).
# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

package(
    default_visibility =
        ["//tensorflow/python/saved_model:__subpackages__"],
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "pywrap_libexport",
    srcs = ["pywrap_libexport.cc"],
    module_name = "pywrap_libexport",
    deps = [
        "//tensorflow/cc/experimental/libexport:constants",
        "//tensorflow/cc/experimental/libexport:save",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)
