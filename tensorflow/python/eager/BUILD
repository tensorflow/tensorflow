licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_py_test", "tf_cc_binary")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_py_logged_benchmark",
)
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")

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
    visibility = [
        "//learning/deepmind/courier:__subpackages__",
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:tape",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python:cpp_python_util",
        "//tensorflow/python:ndarray_tensor",
        "//tensorflow/python:ndarray_tensor_bridge",
        "//tensorflow/python:numpy_lib",
        "//tensorflow/python:py_seq_tensor",
        "//tensorflow/python:safe_ptr",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
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
        ":def_function",
        ":execute",
        ":execution_callbacks",
        ":function",
        ":graph_only_ops",
        ":monitoring",
        ":profiler",
        ":profiler_client",
        ":tape",
        ":test",
        ":wrap_function",
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
        "//tensorflow/python:device_spec",
        "//tensorflow/python:errors",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:tf2",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "monitoring",
    srcs = ["monitoring.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

cuda_py_test(
    name = "monitoring_test",
    srcs = ["monitoring_test.py"],
    additional_deps = [
        ":monitoring",
        ":test",
    ],
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "profiler",
    srcs = ["profiler.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "profiler_test",
    srcs = ["profiler_test.py"],
    additional_deps = [
        ":profiler",
        ":test",
        "//tensorflow/python:constant_op",
        "//tensorflow/core/profiler:protos_all_py",
    ],
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "profiler_client",
    srcs = ["profiler_client.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tensorflow",
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
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "backprop_test",
    srcs = ["backprop_test.py"],
    additional_deps = [
        ":backprop",
        ":context",
        ":test",
        "//third_party/py/numpy",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:training",
    ],
    tags = ["no_rocm"],
    xla_enable_strict_auto_jit = True,
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
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "function_argument_naming_test",
    size = "medium",
    srcs = ["function_argument_naming_test.py"],
    additional_deps = [
        ":backprop",
        ":def_function",
        ":function",
        ":test",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:math_ops",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "function_defun_collection_test",
    size = "medium",
    srcs = ["function_defun_collection_test.py"],
    additional_deps = [
        ":backprop",
        ":def_function",
        ":function",
        ":test",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "function_gradients_test",
    size = "medium",
    srcs = ["function_gradients_test.py"],
    additional_deps = [
        ":backprop",
        ":context",
        ":def_function",
        ":function",
        ":test",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
    ],
    shard_count = 5,
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "function_test",
    size = "medium",
    srcs = ["function_test.py"],
    additional_deps = [
        ":backprop",
        ":context",
        ":def_function",
        ":function",
        ":test",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:test_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
    ],
    shard_count = 15,
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

tf_py_test(
    name = "execution_callbacks_test",
    srcs = ["execution_callbacks_test.py"],
    additional_deps = [
        ":execution_callbacks",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
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
    xla_enable_strict_auto_jit = True,
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
        "//tensorflow/python:func_graph",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:graph_to_function_def",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:tape",
        "//third_party/py/numpy",
        "@six_archive//:six",
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
        "//tensorflow/python:unconnected_gradients",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
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
        ":profiler",
        "//third_party/py/numpy",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/keras",
    ],
    xla_enable_strict_auto_jit = True,
)

tf_py_logged_benchmark(
    name = "benchmarks",
    target = "//tensorflow/python/eager:benchmarks_test",
)

tf_py_test(
    name = "tape_test",
    srcs = ["tape_test.py"],
    additional_deps = [
        ":backprop",
        ":context",
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
        "//tensorflow/python:config",
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
    xla_enable_strict_auto_jit = True,
)

tf_py_test(
    name = "pywrap_tfe_test",
    srcs = ["pywrap_tfe_test.py"],
    additional_deps = [
        ":backprop",
        ":context",
        ":core",
        ":test",
        "//third_party/py/numpy",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:random_ops",
    ],
)

py_library(
    name = "imperative_grad",
    srcs = ["imperative_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:unconnected_gradients",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "memory_test",
    size = "medium",
    srcs = ["memory_test.py"],
    additional_deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/keras",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "@six_archive//:six",
    ],
    tags = [
        "optonly",  # The test is too slow in non-opt mode
    ],
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "def_function",
    srcs = ["def_function.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":function",
        ":lift_to_graph",
        "//tensorflow/python:cond_v2",  # TODO(b/118513001): Imported via control_flow_ops; remove.
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:while_v2",  # TODO(b/118513001): Imported via control_flow_ops; remove.
        "//tensorflow/python/training/tracking:base",
    ],
)

py_library(
    name = "lift_to_graph",
    srcs = ["lift_to_graph.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        "//tensorflow/python:framework_ops",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "lift_to_graph_test",
    size = "medium",
    srcs = ["lift_to_graph_test.py"],
    additional_deps = [
        "lift_to_graph",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:framework_ops",
    ],
)

tf_py_test(
    name = "def_function_test",
    srcs = ["def_function_test.py"],
    additional_deps = [
        ":def_function",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
    ],
)

tf_xla_py_test(
    name = "def_function_xla_test",
    srcs = ["def_function_xla_test.py"],
    tags = [
        "no_pip",
        "no_rocm",
        "nomac",
    ],
    deps = [
        ":def_function",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
    ],
)

py_library(
    name = "wrap_function",
    srcs = ["wrap_function.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":function",
        ":lift_to_graph",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/training/tracking:base",
    ],
)

tf_py_test(
    name = "wrap_function_test",
    srcs = ["wrap_function_test.py"],
    additional_deps = [
        ":wrap_function",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
    ],
)

py_library(
    name = "remote",
    srcs = ["remote.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python/eager:context",
    ],
)
