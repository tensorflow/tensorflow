# Description:
# Python APIs for various Tensorflow backends.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "compiler",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/compiler/mlir",
        "//tensorflow/python/compiler/tensorrt:init_py",
        "//tensorflow/python/compiler/xla:compiler_py",
    ],
)
