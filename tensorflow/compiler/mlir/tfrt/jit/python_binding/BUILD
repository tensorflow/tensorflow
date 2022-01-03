load("//tensorflow:tensorflow.bzl", "pybind_extension")
load("//tensorflow:tensorflow.bzl", "py_strict_test")

licenses(["notice"])

package(
    default_visibility = [":__subpackages__"],
)

py_library(
    name = "tf_cpurt",
    testonly = 1,
    srcs = ["tf_cpurt.py"],
    visibility = ["//tensorflow/compiler/mlir/tfrt:__subpackages__"],
    deps = [
        ":_tf_cpurt_executor",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_cpurt_test",
    srcs = ["tf_cpurt_test.py"],
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":tf_cpurt",
        # copybara:uncomment "//testing/pybase",
        "//third_party/py/numpy",
    ],
)

pybind_extension(
    name = "_tf_cpurt_executor",
    srcs = ["tf_cpurt_executor.cc"],
    hdrs = ["tf_cpurt_executor.h"],
    module_name = "_tf_cpurt_executor",
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:tf_cpurt_pipeline",
        "//tensorflow/compiler/mlir/tfrt/python_tests:python_test_attrs_registration",
        "//tensorflow/core/platform:dynamic_annotations",
        "//third_party/eigen3",
        "//third_party/python_runtime:headers",  # build_cleaner: keep
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@pybind11",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//backends/cpu:cpurt",
    ],
)
