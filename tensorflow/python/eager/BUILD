load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")
load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_py_logged_benchmark",
)
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "pywrap_tfe_lib",
    srcs = [
        "pywrap_tensor.cc",
        "pywrap_tensor_conversion.cc",
        "pywrap_tfe_src.cc",
    ],
    hdrs = [
        "pywrap_tensor.h",
        "pywrap_tensor_conversion.h",
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
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/python:cpp_python_util",
        "//tensorflow/python:ndarray_tensor",
        "//tensorflow/python:ndarray_tensor_bridge",
        "//tensorflow/python:numpy_lib",
        "//tensorflow/python:py_seq_tensor",
        "//tensorflow/python:safe_ptr",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
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
        ":cancellation",
        ":context",
        ":core",
        ":def_function",
        ":execute",
        ":forwardprop",
        ":forwardprop_util",
        ":function",
        ":graph_only_ops",
        ":monitoring",
        ":profiler",
        ":profiler_client",
        ":tape",
        ":test",
        ":wrap_function",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/eager/memory_tests:memory_test_util",
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
    name = "cancellation",
    srcs = ["cancellation.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

cuda_py_test(
    name = "cancellation_test",
    size = "small",
    srcs = ["cancellation_test.py"],
    additional_deps = [
        ":cancellation",
        ":test",
    ],
)

py_library(
    name = "executor",
    srcs = ["executor.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

py_library(
    name = "context",
    srcs = ["context.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":executor",
        ":monitoring",
        "//tensorflow/python:device",
        "//tensorflow/python:device_spec",
        "//tensorflow/python:errors",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:tf2",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "context_test",
    size = "small",
    srcs = ["context_test.py"],
    additional_deps = [
        ":context",
        ":test",
    ],
)

py_library(
    name = "monitoring",
    srcs = ["monitoring.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_agents:__subpackages__",
    ],
    deps = [
        "//tensorflow/python:c_api_util",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "monitoring_test",
    srcs = ["monitoring_test.py"],
    additional_deps = [
        ":monitoring",
        ":test",
    ],
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
    ],
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

py_test(
    name = "profiler_client_test",
    srcs = ["profiler_client_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":profiler_client",
        ":test",
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
    tags = [
        "no_rocm",
        "no_windows",  #TODO(b/139745667)
    ],
)

cuda_py_test(
    name = "forwardprop_test",
    srcs = ["forwardprop_test.py"],
    additional_deps = [
        ":forwardprop",
        ":forwardprop_util",
        ":test",
        "//tensorflow/python/distribute:mirrored_strategy",
    ],
    shard_count = 5,
)

cuda_py_test(
    name = "core_test",
    size = "small",
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
)

cuda_py_test(
    name = "function_test",
    size = "medium",
    srcs = ["function_test.py"],
    additional_deps = [
        ":backprop",
        ":cancellation",
        ":context",
        ":def_function",
        ":function",
        ":test",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:gradients",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:indexed_slices",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:test_ops",
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
        ":execute",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:tensor_shape",
    ],
)

py_library(
    name = "function",
    srcs = ["function.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":core",
        ":execute",
        ":forwardprop_util",
        ":graph_only_ops",
        ":tape",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:graph_to_function_def",
        "//tensorflow/python:util",
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
        ":backprop_util",
        ":context",
        ":execute",
        ":imperative_grad",
        ":tape",
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
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "@six_archive//:six",
    ],
)

py_library(
    name = "backprop_util",
    srcs = ["backprop_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:tensor_util",
    ],
)

py_library(
    name = "forwardprop",
    srcs = ["forwardprop.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":forwardprop_util",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "forwardprop_util",
    srcs = ["forwardprop_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

cuda_py_test(
    name = "benchmarks_test",
    srcs = ["benchmarks_test.py"],
    additional_deps = [
        ":backprop",
        ":context",
        ":forwardprop",
        ":function",
        ":test",
        ":profiler",
        ":remote",
        "//third_party/py/numpy",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/keras",
    ],
)

cuda_py_test(
    name = "remote_benchmarks_test",
    srcs = ["remote_benchmarks_test.py"],
    additional_deps = [
        ":backprop",
        ":context",
        ":forwardprop",
        ":function",
        ":test",
        ":profiler",
        ":remote",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:random_ops",
    ],
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
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:op_selector",
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
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "def_function_test",
    srcs = ["def_function_test.py"],
    additional_deps = [
        ":def_function",
        "//tensorflow/python/autograph/core",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
    ],
)

cuda_py_test(
    name = "def_function_xla_jit_test",
    srcs = ["def_function_xla_jit_test.py"],
    additional_deps = [
        ":backprop",
        ":def_function",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
    ],
    tags = [
        "no_mac",
        "no_windows",
    ],
    xla_enabled = True,
)

tf_xla_py_test(
    name = "def_function_xla_test",
    srcs = ["def_function_xla_test.py"],
    tags = [
        "no_pip",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":def_function",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
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
        "//tensorflow/python/saved_model:nested_structure_coder",
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
        ":context",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
    ],
)

cuda_py_test(
    name = "remote_test",
    size = "medium",
    srcs = ["remote_test.py"],
    additional_deps = [
        ":context",
        ":def_function",
        ":test",
        ":remote",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
        "//tensorflow/python:resource_variable_ops",
    ],
    grpc_enabled = True,
    shard_count = 2,
    tags = [
        "no_oss",  # This test launches local server.
        "optonly",  # times out
    ],
)

tpu_py_test(
    name = "remote_cloud_tpu_test",
    srcs = ["remote_cloud_tpu_test.py"],
    tags = [
        "notap",
    ],
    deps = [
        ":context",
        ":remote",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/tpu:tpu_strategy_util",
    ],
)

cuda_py_test(
    name = "device_placement_test",
    size = "small",
    srcs = ["device_placement_test.py"],
    additional_deps = [
        ":context",
        ":def_function",
        ":test",
        ":remote",
        "//tensorflow/python:resource_variable_ops",
    ],
)
