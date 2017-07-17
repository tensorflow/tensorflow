# TensorFlow shared library for training decision trees.
package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["*"],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "init_py",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_py",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_py",
    ],
)

# Pip

py_library(
    name = "decision_trees_pip",
    deps = [
        ":init_py",
    ],
)
