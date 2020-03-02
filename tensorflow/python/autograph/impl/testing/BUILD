load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "pybind_for_testing",
    srcs = ["pybind_for_testing.cc"],
    module_name = "pybind_for_testing",
    deps = [
        "@pybind11",
    ],
)
