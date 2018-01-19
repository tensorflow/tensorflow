licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_py_logged_benchmark",
)

cc_library(
    name = "pywrap_tfe_lib",
    srcs = [
        "pywrap_tensor.cc",
        "pywrap_tfe_src.cc",
    ],
    hdrs = [
        "pywrap_tensor.h",
        "pywrap_tfe.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:tape",
        "//tensorflow/core:lib",
        "//tensorflow/python:ndarray_tensor",
        "//tensorflow/python:ndarray_tensor_bridge",
        "//tensorflow/python:numpy_lib",
        "//tensorflow/python:py_seq_tensor",
        "//tensorflow/python:safe_ptr",
        "//util/python:python_headers",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "eager_pip",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":backprop",
        ":context",
        ":core",
        ":custom_gradient",
        ":execute",
        ":function",
        ":graph_callable",
        ":graph_only_ops",
        ":tape",
        ":test",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        "//tensorflow/python:errors",
        "//tensorflow/python:pywrap_tensorflow",
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
)

cuda_py_test(
    name = "tensor_test",
    srcs = ["tensor_test.py"],
    additional_deps = [
        ":context",
        ":test",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
    ],
)

cuda_py_test(
    name = "backprop_test",
    srcs = ["backprop_test.py"],
    additional_deps = [
        ":backprop",
        ":context",
        ":custom_gradient",
        ":test",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:training",
    ],
)

cuda_py_test(
    name = "core_test",
    srcs = ["core_test.py"],
    additional_deps = [
        ":context",
        ":core",
        ":execute",
        ":test",
        "//third_party/py/numpy",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

cuda_py_test(
    name = "function_test",
    srcs = ["function_test.py"],
    additional_deps = [
        ":backprop",
        ":context",
        ":function",
        ":tape",
        ":test",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
    ],
)

py_library(
    name = "test",
    srcs = ["test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":core",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
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
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "execution_callbacks",
    srcs = ["execution_callbacks.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        "//tensorflow/python:pywrap_tensorflow",
        "//third_party/py/numpy",
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
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_binary(
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
        ":context",
        ":tape",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
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
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:tape",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "graph_callable",
    srcs = ["graph_callable.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:tape",
    ],
)

py_test(
    name = "graph_callable_test",
    srcs = ["graph_callable_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":backprop",
        ":graph_callable",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:function",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "backprop",
    srcs = ["backprop.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":imperative_grad",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:tape",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "benchmarks_test",
    srcs = ["benchmarks_test.py"],
    additional_deps = [
        ":backprop",
        ":context",
        ":function",
        ":test",
        "//third_party/py/numpy",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:random_ops",
    ],
)

tf_py_logged_benchmark(
    name = "benchmarks",
    target = "//tensorflow/python/eager:benchmarks_test",
)

py_test(
    name = "tape_test",
    srcs = ["tape_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":backprop",
        ":context",
        ":custom_gradient",
        ":test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "ops_test",
    srcs = ["ops_test.py"],
    additional_deps = [
        ":context",
        ":execute",
        ":test",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:tensor_shape",
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

py_library(
    name = "imperative_grad",
    srcs = ["imperative_grad.py"],
    srcs_version = "PY2AND3",
)
