load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],
)

pybind_extension(
    name = "pybind_for_testing",
    srcs = ["pybind_for_testing.cc"],
    deps = [
        "@pybind11",
    ],
)
