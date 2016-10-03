licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "tfprof_logger",
    srcs = ["tfprof_logger.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/tfprof/tools/tfprof:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "tfprof_logger_test",
    srcs = ["tfprof_logger_test.py"],
    additional_deps = [
        ":tfprof_logger",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tfprof/tools/tfprof:protos_all_py",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

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
