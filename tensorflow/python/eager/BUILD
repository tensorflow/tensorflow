licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

cc_library(
    name = "pywrap_tfe_lib",
    srcs = ["pywrap_tfe_src.cc"],
    hdrs = ["pywrap_tfe.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/core:lib",
        "//tensorflow/python:numpy_lib",
        "//tensorflow/python:py_func_lib",
        "//util/python:python_headers",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":memory_trace",
        ":tape",
        "//tensorflow/python:errors",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

py_library(
    name = "tensor",
    srcs = ["tensor.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":core",
        ":tape",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:tensor_shape",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "context",
    srcs = ["context.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:device",
        "//tensorflow/python:errors",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "tape",
    srcs = ["tape.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "memory_trace",
    srcs = ["memory_trace.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "tensor_test",
    srcs = ["tensor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensor",
        ":test",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "core_test",
    srcs = ["core_test.py"],
    additional_deps = [
        ":context",
        ":core",
        ":execute",
        ":tensor",
        ":test",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_library(
    name = "test",
    srcs = ["test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "execute",
    srcs = ["execute.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":core",
        ":tape",
        ":tensor",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "@six_archive//:six",
    ],
)

cc_library(
    name = "python_eager_op_gen",
    srcs = ["python_eager_op_gen.cc"],
    hdrs = ["python_eager_op_gen.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python:python_op_gen",
    ],
)

cc_library(
    name = "python_eager_op_gen_main",
    srcs = [
        "python_eager_op_gen_main.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":python_eager_op_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_binary(
    name = "python_eager_op_gen_demo",
    deps = [
        ":python_eager_op_gen_main",
        "//tensorflow/core:ops",
    ],
)

py_library(
    name = "custom_gradient",
    srcs = ["custom_gradient.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":core",
        ":tape",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "graph_only_ops",
    srcs = ["graph_only_ops.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
    ],
)

cuda_py_test(
    name = "graph_only_ops_test",
    srcs = ["graph_only_ops_test.py"],
    additional_deps = [
        "graph_only_ops",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "framework_for_generated_wrappers",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/eager:execute",
    ],
)

py_library(
    name = "function",
    srcs = ["function.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":graph_only_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:graph_to_function_def",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/eager:tensor",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "pip_dependencies",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":core",
        ":execute",
        ":graph_only_ops",
        ":tensor",
        ":test",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets.

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
