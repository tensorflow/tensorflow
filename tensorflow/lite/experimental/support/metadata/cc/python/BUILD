load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = [
        "//tensorflow/lite/experimental/support/metadata:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "_pywrap_metadata_version",
    srcs = [
        "metadata_version.cc",
    ],
    features = ["-use_header_modules"],
    module_name = "_pywrap_metadata_version",
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/support/metadata/cc:metadata_version",
        "@pybind11",
    ],
)
