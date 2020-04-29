load("//tensorflow:tensorflow.bzl", "py_strict_library")

package(
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

# Important: this is a leaf library. It may not have any new dependencies inside TF proper.
# The sole exception is tf_export, to allow exporting symbols into the public namespace.
py_strict_library(
    name = "types",
    srcs = [
        "__init__.py",
        "core.py",
        "internal.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [],
)
