load("//tensorflow:tensorflow.bzl", "pytype_strict_library")

package(
    licenses = ["notice"],
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
    visibility = ["//visibility:private"],
)

# Important: this is a leaf library. It may not have any new dependencies inside TF proper.
# The sole exception is tf_export, to allow exporting symbols into the public namespace.
pytype_strict_library(
    name = "types",
    srcs = [
        "__init__.py",
        "core.py",
        "distribute.py",
        "internal.py",
        "trace.py",
    ],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:types_whitelist",
    ],
    deps = [
        ":doc_typealias",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
        "//third_party/py/numpy",
        "@typing_extensions_archive//:typing_extensions",
    ],
)

pytype_strict_library(
    name = "doc_typealias",
    srcs = [
        "doc_typealias.py",
    ],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
    deps = [],
)
