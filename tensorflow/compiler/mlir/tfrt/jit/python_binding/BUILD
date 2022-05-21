# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pybind_library")
load("//tensorflow:tensorflow.bzl", "py_strict_test")

licenses(["notice"])

package(
    default_visibility = [":__subpackages__"],
)

py_library(
    name = "tf_jitrt",
    testonly = 1,
    srcs = ["tf_jitrt.py"],
    visibility = ["//tensorflow/compiler/mlir/tfrt:__subpackages__"],
    deps = [
        ":_tf_jitrt_executor",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_jitrt_test",
    srcs = ["tf_jitrt_test.py"],
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":tf_jitrt",
        # copybara:uncomment "//testing/pybase",
        "//third_party/py/numpy",
    ],
)

pybind_extension(
    name = "_tf_jitrt_executor",
    srcs = ["tf_jitrt_executor.cc"],
    hdrs = ["tf_jitrt_executor.h"],
    deps = [
        ":conversion_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:tf_jitrt_pipeline",
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
        "@tf_runtime//backends/jitrt",
        "@tf_runtime//backends/jitrt:jitrt_compiler",
    ],
)

py_library(
    name = "tfrt_fallback",
    testonly = True,
    srcs = ["tfrt_fallback.py"],
    visibility = ["//tensorflow/compiler/mlir/tfrt:__subpackages__"],
    deps = [
        ":_tfrt_fallback",
    ],
)

pybind_extension(
    name = "_tfrt_fallback",
    testonly = True,
    srcs = ["tfrt_fallback.cc"],
    hdrs = ["tfrt_fallback.h"],
    deps = [
        ":conversion_utils",
        "//tensorflow/compiler/mlir/tfrt:runtime_fallback_executor",
        "//tensorflow/compiler/mlir/tfrt:tf_jitrt",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/runtime_fallback/util:type_util",
        "//third_party/python_runtime:headers",  # build_cleaner: keep
        "@llvm-project//llvm:Support",
        "@pybind11",
        "@tf_runtime//:dtype",
    ],
)

pybind_library(
    name = "conversion_utils",
    srcs = ["conversion_utils.cc"],
    hdrs = ["conversion_utils.h"],
    deps = [
        "@tf_runtime//:dtype",
        "@tf_runtime//backends/jitrt:types",
    ],
)
