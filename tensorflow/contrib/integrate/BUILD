# Description:
#   Integration and ODE solvers for TensorFlow.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "integrate_py",
    srcs = [
        "__init__.py",
        "python/ops/odes.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "odes_test",
    srcs = ["python/ops/odes_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":integrate_py",
        "//tensorflow:tensorflow_py",
    ],
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
)
