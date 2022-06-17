load("//tensorflow:tensorflow.bzl", "check_deps")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pybind_extension")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")
load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_py_logged_benchmark",
)
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_rpc_deps",
)

package(
    licenses = ["notice"],
)

cc_library(
    name = "pywrap_tfe_lib",
    srcs = [
        "pywrap_gradient_exclusions.cc",
        "pywrap_tensor.cc",
        "pywrap_tensor_conversion.cc",
        "pywrap_tfe_src.cc",
    ],
    hdrs = [
        "pywrap_gradient_exclusions.h",
        "pywrap_tensor.h",
        "pywrap_tensor_conversion.h",
        "pywrap_tfe.h",
    ],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",  # Required for pybind11
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:dlpack",
        "//tensorflow/c/eager:tape",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/c/eager:tfe_op_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/util:managed_stack_trace",
        "//tensorflow/python:ndarray_tensor",
        "//tensorflow/python:ndarray_tensor_bridge",
        "//tensorflow/python:numpy_lib",
        "//tensorflow/python:py_exception_registry",
        "//tensorflow/python:pybind11_status",
        "//tensorflow/python/lib/core:py_seq_tensor",
        "//tensorflow/python/lib/core:py_util",
        "//tensorflow/python/lib/core:safe_ptr",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "//tensorflow/python/util:stack_trace",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_tensor_test_util",
    testonly = True,
    srcs = ["pywrap_tensor_test_util.cc"],
    deps = [
        ":pywrap_tfe_lib",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/python:pybind11_lib",
        "@pybind11",
    ],
)

cuda_py_test(
    name = "pywrap_tensor_test",
    size = "small",
    srcs = ["pywrap_tensor_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/168051787): Enable.
        "no_pip",  # TODO(b/168051787): Enable.
    ],
    deps = [
        ":pywrap_tensor_test_util",
        ":test",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "pywrap_tensor.h",
        "pywrap_tensor_conversion.h",
        "pywrap_tfe.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "eager_pip",
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":backprop",
        ":benchmarks_test_base",
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
        "//tensorflow/python/dlpack",
        "//tensorflow/python/eager/memory_tests:memory_test_util",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        "//tensorflow/python:errors",
        "//tensorflow/python:pywrap_tfe",
    ],
)

py_library(
    name = "cancellation",
    srcs = ["cancellation.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tfe",
    ],
)

cuda_py_test(
    name = "cancellation_test",
    size = "small",
    srcs = ["cancellation_test.py"],
    python_version = "PY3",
    deps = [
        ":cancellation",
        ":test",
    ],
)

py_library(
    name = "executor",
    srcs = ["executor.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tfe",
    ],
)

py_library(
    name = "context",
    srcs = ["context.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":executor",
        ":monitoring",
        "//tensorflow/python:c_api_util",
        "//tensorflow/python:device",
        "//tensorflow/python:device_spec",
        "//tensorflow/python:errors",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:tf2",
        "//tensorflow/python:tfrt_utils",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

tf_python_pybind_extension(
    name = "custom_device_testutil",
    testonly = True,
    srcs = ["custom_device_testutil.cc"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:custom_device_testutil",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_test(
    name = "custom_device_test",
    size = "small",
    srcs = ["custom_device_test.py"],
    python_version = "PY3",
    # Note that this currently only works with --config=monolithic, since it
    # requires the C API which runs static initializers again.
    #
    # TODO(allenl): Figure out a way to allow extensions to register custom
    # devices which works with dynamic linking.
    tags = [
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":context",
        ":custom_device_testutil",
        ":test",
    ],
)

cuda_py_test(
    name = "context_test",
    size = "small",
    srcs = ["context_test.py"],
    python_version = "PY3",
    xla_enabled = True,
    deps = [
        ":context",
        ":test",
        "//tensorflow/compiler/xla/service:hlo_proto_py",
    ],
)

py_library(
    name = "monitoring",
    srcs = ["monitoring.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:c_api_util",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "monitoring_test",
    srcs = ["monitoring_test.py"],
    python_version = "PY3",
    deps = [
        ":monitoring",
        ":test",
    ],
)

py_library(
    name = "profiler",
    srcs = ["profiler.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        "//tensorflow/python:util",
        "//tensorflow/python/profiler/internal:_pywrap_profiler",
    ],
)

cuda_py_test(
    name = "profiler_test",
    srcs = ["profiler_test.py"],
    python_version = "PY3",
    tags = [
        "no_rocm",
        "no_windows",  # TODO(b/184424727): Re-enable this.
    ],
    deps = [
        ":test",
        "//tensorflow/core/profiler/protobuf:trace_events_proto_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python/profiler:profiler_v2",
        "//tensorflow/python/profiler:trace",
    ],
)

py_library(
    name = "profiler_client",
    srcs = ["profiler_client.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/profiler/internal:_pywrap_profiler",
    ],
)

py_test(
    name = "profiler_client_test",
    srcs = ["profiler_client_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",
    ],
    deps = [
        ":profiler_client",
        ":test",
    ],
)

py_library(
    name = "tape",
    srcs = ["tape.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
)

cuda_py_test(
    name = "tensor_test",
    srcs = ["tensor_test.py"],
    python_version = "PY3",
    deps = [
        ":context",
        ":test",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
    ],
)

cuda_py_test(
    name = "backprop_test",
    srcs = ["backprop_test.py"],
    python_version = "PY3",
    tags = [
        "no_cuda_asan",  # b/173825938
        "no_windows",  #TODO(b/139745667)
        "notsan",  #TODO(b/139745667)
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":backprop",
        ":context",
        ":test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/framework:memory_checker",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "forwardprop_test",
    srcs = ["forwardprop_test.py"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":forwardprop",
        ":forwardprop_util",
        ":test",
        "//tensorflow/python/distribute:mirrored_strategy",
    ],
)

cuda_py_test(
    name = "core_test",
    size = "small",
    srcs = ["core_test.py"],
    python_version = "PY3",
    tags = [
        "notsan",  # TODO(b/183962087)
    ],
    deps = [
        ":context",
        ":core",
        ":execute",
        ":test",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:pywrap_tfe",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "function_argument_naming_test",
    size = "medium",
    srcs = ["function_argument_naming_test.py"],
    python_version = "PY3",
    # TODO(b/169371527): insert transfer op in eager lowering for TFRT.
    deps = [
        ":backprop",
        ":def_function",
        ":function",
        ":test",
        "//tensorflow/python:math_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "function_defun_collection_test",
    size = "medium",
    srcs = ["function_defun_collection_test.py"],
    python_version = "PY3",
    deps = [
        ":backprop",
        ":def_function",
        ":function",
        ":test",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "function_gradients_test",
    size = "medium",
    srcs = ["function_gradients_test.py"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":backprop",
        ":context",
        ":def_function",
        ":function",
        ":test",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "function_test",
    size = "medium",
    srcs = ["function_test.py"],
    python_version = "PY3",
    shard_count = 15,
    tags = [
        "nomac",  # b/157056289
    ],
    deps = [
        ":backprop",
        ":cancellation",
        ":context",
        ":def_function",
        ":function",
        ":test",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:indexed_slices",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sendrecv_ops_gen",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:test_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "test",
    srcs = ["test.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":core",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "graph_only_ops",
    srcs = ["graph_only_ops.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:op_callbacks",
        "//tensorflow/python:tensor_shape",
    ],
)

cuda_py_test(
    name = "graph_only_ops_test",
    srcs = ["graph_only_ops_test.py"],
    python_version = "PY3",
    deps = [
        "graph_only_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "framework_for_generated_wrappers",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":execute",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:tensor_shape",
    ],
)

py_library(
    name = "function_spec",
    srcs = ["function_spec.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_decorator",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "function",
    srcs = ["function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":core",
        ":execute",
        ":forwardprop_util",
        ":function_context",
        ":function_saved_model_utils",
        ":function_spec",
        ":graph_only_ops",
        ":tape",
        "//tensorflow/core/function/polymorphism:function_cache",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:graph_to_function_def",
        "//tensorflow/python:handle_data_util",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python:util",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "function_context",
    srcs = ["function_context.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        "//tensorflow/core/function/polymorphism:function_cache",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/saved_model:save_context",
    ],
)

py_library(
    name = "function_saved_model_utils",
    srcs = ["function_saved_model_utils.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:resource",
    ],
)

pybind_extension(
    name = "_concrete_function",
    srcs = ["function.cc"],
    deps = [
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_library(
    name = "backprop",
    srcs = ["backprop.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":backprop_util",
        ":context",
        ":execute",
        ":imperative_grad",
        ":tape",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:unconnected_gradients",
        "//tensorflow/python:util",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "@six_archive//:six",
    ],
)

py_library(
    name = "backprop_util",
    srcs = ["backprop_util.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:tensor_util",
    ],
)

py_library(
    name = "forwardprop",
    srcs = ["forwardprop.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":forwardprop_util",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:util",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
    ],
)

py_library(
    name = "forwardprop_util",
    srcs = ["forwardprop_util.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tfe",
    ],
)

py_library(
    name = "benchmarks_test_base",
    srcs = ["benchmarks_test_base.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [":test"],
)

cuda_py_test(
    name = "benchmarks_test",
    srcs = ["benchmarks_test.py"],
    python_version = "PY3",
    deps = [
        ":backprop",
        ":benchmarks_test_base",
        ":context",
        ":forwardprop",
        ":function",
        ":remote",
        ":test",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "run_eager_op_as_function_test",
    srcs = ["run_eager_op_as_function_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",  # b/207695287
    ],
    deps = [
        ":benchmarks_test_base",
        ":context",
        ":test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/data",
    ],
)

tf_xla_py_test(
    name = "run_eager_op_as_function_xla_test",
    size = "small",
    srcs = ["run_eager_op_as_function_xla_test.py"],
    enable_mlir_bridge = False,
    enabled_backends = [
        "cpu",
        "gpu",
    ],
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":context",
        ":test",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "remote_benchmarks_test",
    srcs = ["remote_benchmarks_test.py"],
    python_version = "PY3",
    deps = [
        ":backprop",
        ":benchmarks_test_base",
        ":context",
        ":forwardprop",
        ":function",
        ":profiler",
        ":remote",
        ":test",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:random_ops",
    ],
)

tf_py_logged_benchmark(
    name = "benchmarks",
    python_version = "PY3",
    target = "//tensorflow/python/eager:benchmarks_test",
)

tf_py_test(
    name = "tape_test",
    srcs = ["tape_test.py"],
    python_version = "PY3",
    deps = [
        ":backprop",
        ":context",
        ":tape",
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
    python_version = "PY3",
    deps = [
        ":context",
        ":execute",
        ":test",
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
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "pywrap_tfe_test",
    srcs = ["pywrap_tfe_test.py"],
    python_version = "PY3",
    deps = [
        ":backprop",
        ":context",
        ":core",
        ":test",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:test_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "imperative_grad",
    srcs = ["imperative_grad.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:unconnected_gradients",
        "//tensorflow/python:util",
    ],
)

# Check that the main TensorFlow python library depends on and links XLA.
check_deps(
    name = "def_function_check_deps",
    required_deps = [
        "//tensorflow/compiler/jit:xla_kernel_creator",
    ],
    deps = [":def_function"],
)

py_library(
    name = "def_function",
    srcs = ["def_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":function",
        ":function_spec",
        ":lift_to_graph",
        "//tensorflow/python:cond_v2",  # TODO(b/118513001): Imported via control_flow_ops; remove.
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:while_v2",  # TODO(b/118513001): Imported via control_flow_ops; remove.
        "//tensorflow/python/distribute/parallel_device",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/trackable:base",
    ],
)

py_library(
    name = "lift_to_graph",
    srcs = ["lift_to_graph.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    deps = [
        "lift_to_graph",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:test",
    ],
)

cuda_py_test(
    name = "def_function_test",
    srcs = ["def_function_test.py"],
    python_version = "PY3",
    # TODO(b/169371527): insert transfer op in eager lowering for TFRT.
    deps = [
        ":def_function",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model:save_options",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "def_function_test_cpu_only",
    srcs = ["def_function_test_cpu_only.py"],
    python_version = "PY3",
    # --config=cuda implicitly links in XLA.
    tags = [
        "no_cuda_on_cpu_tap",
        "no_gpu",
        "no_oss",  # No way to force no XLA linkage in OSS build from here.
        "no_pip",
    ],
    deps = [
        ":def_function",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/autograph/core",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "def_function_xla_jit_test",
    srcs = ["def_function_xla_jit_test.py"],
    disabled_backends = [
        "cpu_ondemand",
    ],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_mac",
        "no_pip",
        "no_tfrt",  # TODO(b/185944215)
        "no_windows",
    ],
    use_xla_device = False,
    deps = [
        ":backprop",
        ":def_function",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
    ],
)

tf_xla_py_test(
    name = "def_function_xla_test",
    srcs = ["def_function_xla_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
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
        # TODO(b/145618471): Remove this transitive dependency.
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python:framework_ops",
    ],
)

py_library(
    name = "wrap_function",
    srcs = ["wrap_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":function",
        ":lift_to_graph",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/trackable:base",
    ],
)

tf_py_test(
    name = "wrap_function_test",
    srcs = ["wrap_function_test.py"],
    python_version = "PY3",
    deps = [
        ":wrap_function",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
    ],
)

cuda_py_test(
    name = "wrap_function_device_test",
    srcs = ["wrap_function_device_test.py"],
    python_version = "PY3",
    xla_enable_strict_auto_jit = False,
    xla_enabled = False,
    deps = [
        ":def_function",
        ":wrap_function",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:config",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "remote",
    srcs = ["remote.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
    ] + tf_additional_rpc_deps(),
)

cuda_py_test(
    name = "remote_test",
    size = "medium",
    srcs = ["remote_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "no_oss",  # This test launches local server.
        "optonly",  # times out
    ],
    deps = [
        ":cancellation",
        ":context",
        ":def_function",
        ":remote",
        ":test",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:test_ops",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "remote_execution_test",
    srcs = ["remote_execution_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_oss",  # This test launches local server
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:remote",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "remote_cluster_test",
    srcs = ["remote_cluster_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_oss",  # This test launches local server
        "no_tfrt",  # TODO(b/171765113)
        "notsan",  # TODO(b/170783249)
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_test(
    name = "remote_cloud_tpu_test",
    srcs = ["remote_cloud_tpu_test.py"],
    python_version = "PY3",
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

tpu_py_test(
    name = "remote_cloud_tpu_pod_test",
    srcs = ["remote_cloud_tpu_test.py"],
    args = ["--num_tpu_devices=32"],
    main = "remote_cloud_tpu_test.py",
    python_version = "PY3",
    tags = [
        "notap",
        "tpu_pod",
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
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":context",
        ":def_function",
        ":remote",
        ":test",
        "//tensorflow/python:resource_variable_ops",
    ],
)

py_library(
    name = "gradient_input_output_exclusions",
    srcs = ["gradient_input_output_exclusions.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "gen_gradient_input_output_exclusions",
    srcs = ["gen_gradient_input_output_exclusions.py"],
    deps = [
        ":gradient_input_output_exclusions",
        "//tensorflow:tensorflow_py",
    ],
)

# Needed for the test below.
exports_files([
    "pywrap_gradient_exclusions.cc",
])

py_test(
    name = "gradient_input_output_exclusions_test",
    srcs = ["gradient_input_output_exclusions_test.py"],
    data = [
        "//tensorflow/python/eager:pywrap_gradient_exclusions.cc",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",  # No point linking the gen script in the pip package.
    ],
    deps = [
        ":gradient_input_output_exclusions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
    ],
)
