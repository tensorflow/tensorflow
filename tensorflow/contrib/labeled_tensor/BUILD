# Description:
#   Labels for TensorFlow.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "labeled_tensor",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":core",
        ":io_ops",
        ":nn",
        ":ops",
        ":sugar",
    ],
)

py_library(
    name = "_typecheck",
    srcs = ["python/ops/_typecheck.py"],
    srcs_version = "PY2AND3",
    visibility = [":__subpackages__"],
)

py_library(
    name = "core",
    srcs = ["python/ops/core.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":_typecheck",
    ],
)

py_library(
    name = "test_util",
    srcs = ["python/ops/test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":_typecheck",
        ":core",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "core_test",
    size = "medium",
    srcs = [
        "python/ops/core_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":core",
        ":test_util",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "io_ops",
    srcs = ["python/ops/io_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":core",
    ],
)

py_test(
    name = "io_ops_test",
    size = "small",
    srcs = [
        "python/ops/io_ops_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":io_ops",
        ":ops",
        ":test_util",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "nn",
    srcs = ["python/ops/nn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":core",
    ],
)

py_test(
    name = "nn_test",
    size = "small",
    srcs = [
        "python/ops/nn_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":nn",
        ":test_util",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "ops",
    srcs = ["python/ops/ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":core",
    ],
)

py_test(
    name = "ops_test",
    size = "medium",
    srcs = [
        "python/ops/ops_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":ops",
        ":test_util",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "sugar",
    srcs = ["python/ops/sugar.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":core",
        ":ops",
    ],
)

py_test(
    name = "sugar_test",
    size = "small",
    srcs = [
        "python/ops/sugar_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":sugar",
        ":test_util",
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
