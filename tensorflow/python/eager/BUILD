load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.bzl", "check_deps")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test", "tf_python_pybind_extension")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_strict_test")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_rpc_deps",
)
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_strict_test")
load(
    "//tensorflow/tools/test:performance.bzl",
    "cuda_py_benchmark_test",
    "tf_py_logged_benchmark",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        "//tensorflow/c:safe_ptr",
        "//tensorflow/c:tf_status_helper",
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
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/util:managed_stack_trace",
        "//tensorflow/python/lib/core:ndarray_tensor",
        "//tensorflow/python/lib/core:ndarray_tensor_bridge",
        "//tensorflow/python/lib/core:py_exception_registry",
        "//tensorflow/python/lib/core:py_seq_tensor",
        "//tensorflow/python/lib/core:py_util",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "//tensorflow/python/util:stack_trace",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_xla//xla/tsl/python/lib/core:numpy",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_tensor_test_util",
    testonly = True,
    srcs = ["pywrap_tensor_test_util.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "pywrap_tensor_test_util.pyi",
    ],
    deps = [
        ":pywrap_tfe_lib",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@pybind11",
    ],
)

cuda_py_strict_test(
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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
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
    visibility = ["//tensorflow/python:__subpackages__"],
)

# Transitive dependencies of this target will be included in the pip package.
py_strict_library(
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

py_strict_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "cancellation",
    srcs = ["cancellation.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tfe",
    ],
)

cuda_py_strict_test(
    name = "cancellation_test",
    size = "small",
    srcs = ["cancellation_test.py"],
    python_version = "PY3",
    deps = [
        ":cancellation",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "executor",
    srcs = ["executor.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tfe",
    ],
)

py_strict_library(
    name = "context",
    srcs = ["context.py"],
    srcs_version = "PY3",
    visibility = [
        "//smartass/brain/ops:__subpackages__",
        "//tensorflow:internal",
        "//third_party/py/jax_tpu_embedding:__subpackages__",
    ],
    deps = [
        ":cancellation",
        ":execute",
        ":executor",
        ":monitoring",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:tf2",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/framework:c_api_util",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:tfrt_utils",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:function_utils",
        "//tensorflow/python/util:is_in_graph_mode",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
    ],
)

tf_python_pybind_extension(
    name = "custom_device_testutil",
    testonly = True,
    srcs = ["custom_device_testutil.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "custom_device_testutil.pyi",
    ],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:safe_ptr",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:custom_device_testutil",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_strict_test(
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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "context_test",
    size = "small",
    srcs = ["context_test.py"],
    python_version = "PY3",
    xla_enabled = True,
    deps = [
        ":context",
        ":def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
        "@local_xla//xla/service:hlo_proto_py",
    ],
)

tpu_py_strict_test(
    name = "context_tpu_platform_test",
    size = "small",
    srcs = ["context_tpu_platform_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
    ],
    deps = [
        ":def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "context_cross_platform_gpu_test",
    size = "small",
    srcs = ["context_cross_platform_gpu_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
    ],
    tpu_ops_enabled = True,
    xla_enabled = True,
    deps = [
        ":def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "context_cross_platform_tpu_test",
    size = "small",
    srcs = ["context_cross_platform_tpu_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
    ],
    deps = [
        ":def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "monitoring",
    srcs = ["monitoring.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/framework:c_api_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "monitoring_test",
    srcs = ["monitoring_test.py"],
    python_version = "PY3",
    deps = [
        ":monitoring",
        ":test",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_strict_library(
    name = "profiler",
    srcs = ["profiler.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        "//tensorflow/python/client:_pywrap_events_writer",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler/internal:_pywrap_profiler",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
    ],
)

cuda_py_strict_test(
    name = "profiler_test",
    srcs = ["profiler_test.py"],
    python_version = "PY3",
    tags = [
        "no_rocm",
        "no_windows",  # TODO(b/184424727): Re-enable this.
    ],
    deps = [
        ":profiler",
        ":test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/profiler:trace",
        "@local_tsl//tsl/profiler/protobuf:trace_events_proto_py",
    ],
)

py_strict_library(
    name = "profiler_client",
    srcs = ["profiler_client.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        "//tensorflow/python/profiler/internal:_pywrap_profiler_plugin",
        "//tensorflow/python/util:deprecation",
    ],
)

py_strict_test(
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
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
    ],
)

cuda_py_strict_test(
    name = "small_constants_optimizer_test",
    size = "medium",
    srcs = ["small_constants_optimizer_test.py"],
    python_version = "PY3",
    deps = [
        ":context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "summary_optimizer_test",
    srcs = ["summary_optimizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/219089812)
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/runtime_client:runtime_client_py",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "@absl_py//absl/flags",
    ],
)

py_strict_library(
    name = "tape",
    srcs = ["tape.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tfe",
    ],
)

py_strict_library(
    name = "record",
    srcs = ["record.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/python:pywrap_tfe"],
)

cuda_py_strict_test(
    name = "tensor_test",
    srcs = ["tensor_test.py"],
    python_version = "PY3",
    deps = [
        ":context",
        ":core",
        ":test",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
    ],
)

cuda_py_strict_test(
    name = "backprop_test",
    srcs = ["backprop_test.py"],
    python_version = "PY3",
    tags = [
        "no_cuda_asan",  # b/173825938
        "no_windows",  #TODO(b/139745667)
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":backprop",
        ":backprop_util",
        ":context",
        ":def_function",
        ":record",
        ":test",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:memory_checker",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:functional_ops",
        "//tensorflow/python/ops:gradient_checker_v2",
        "//tensorflow/python/ops:gradients",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/training",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "forwardprop_test",
    srcs = ["forwardprop_test.py"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":backprop",
        ":context",
        ":def_function",
        ":forwardprop",
        ":forwardprop_util",
        ":record",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:gradient_checker_v2",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_impl",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:unconnected_gradients",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "core_test",
    size = "small",
    srcs = ["core_test.py"],
    python_version = "PY3",
    deps = [
        ":context",
        ":core",
        ":def_function",
        ":execute",
        ":executor",
        ":test",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:resource_variable_ops_gen",
        "//tensorflow/python/ops:script_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "test",
    srcs = ["test.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "execute",
    srcs = ["execute.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":core",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_library(
    name = "graph_only_ops",
    srcs = ["graph_only_ops.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:op_callbacks",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
    ],
)

cuda_py_strict_test(
    name = "graph_only_ops_test",
    srcs = ["graph_only_ops_test.py"],
    python_version = "PY3",
    deps = [
        "graph_only_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "framework_for_generated_wrappers",
    deprecation = "Depending on this target can cause build dependency cycles. Depend on the fine-grained sub-targets instead.",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":execute",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_shape",
    ],
)

py_strict_library(
    name = "function",
    srcs = ["function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/eager/polymorphic_function:atomic_function",
        "//tensorflow/python/eager/polymorphic_function:concrete_function",
        "//tensorflow/python/eager/polymorphic_function:tf_method_target",
        "//tensorflow/python/eager/polymorphic_function:tracing_compilation",
        "//tensorflow/python/eager/polymorphic_function:transform",
    ],
)

py_strict_library(
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
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:composite_tensor_gradient",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:default_gradient",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:unconnected_gradients",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/python/util:variable_utils",
    ],
)

py_strict_library(
    name = "backprop_util",
    srcs = ["backprop_util.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:handle_data_util",
        "//tensorflow/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "forwardprop",
    srcs = ["forwardprop.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":backprop",
        ":backprop_util",
        ":execute",
        ":forwardprop_util",
        "//tensorflow/core/function/polymorphism:function_cache",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/eager/polymorphic_function:tracing_compilation",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:unconnected_gradients",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "forwardprop_util",
    srcs = ["forwardprop_util.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tfe",
    ],
)

py_strict_library(
    name = "benchmarks_test_base",
    srcs = ["benchmarks_test_base.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":test",
        "//tensorflow/python/platform:flags",
    ],
)

cuda_py_benchmark_test(
    name = "benchmarks_test",
    srcs = ["benchmarks_test.py"],
    python_version = "PY3",
    deps = [
        ":backprop",
        ":benchmarks_test_base",
        ":context",
        ":core",
        ":def_function",
        ":forwardprop",
        ":test",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/compat",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:functional_ops",
        "//tensorflow/python/ops:gradients",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_inspect",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
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
        "//tensorflow/python/data/experimental/ops:prefetching_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:bitwise_ops",
        "//tensorflow/python/ops:critical_section_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_map_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/util:tf_inspect",
    ],
)

tf_xla_py_strict_test(
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
        ":def_function",
        ":test",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
    ],
)

cuda_py_benchmark_test(
    name = "remote_benchmarks_test",
    srcs = ["remote_benchmarks_test.py"],
    python_version = "PY3",
    deps = [
        ":context",
        ":def_function",
        ":remote",
        ":test",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/training:server_lib",
    ],
)

tf_py_logged_benchmark(
    name = "benchmarks",
    python_version = "PY3",
    target = "//tensorflow/python/eager:benchmarks_test",
)

tf_py_strict_test(
    name = "record_test",
    srcs = ["record_test.py"],
    python_version = "PY3",
    deps = [
        ":backprop",
        ":context",
        ":record",
        ":test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:variables",
    ],
)

cuda_py_strict_test(
    name = "ops_test",
    srcs = ["ops_test.py"],
    python_version = "PY3",
    deps = [
        ":context",
        ":execute",
        ":test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "pywrap_tfe_test",
    srcs = ["pywrap_tfe_test.py"],
    python_version = "PY3",
    deps = [
        ":backprop",
        ":context",
        ":core",
        ":def_function",
        ":test",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:test_ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "imperative_grad",
    srcs = ["imperative_grad.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/ops:unconnected_gradients",
        "//tensorflow/python/util:compat",
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

py_strict_library(
    name = "def_function",
    srcs = ["def_function.py"],
    srcs_version = "PY3",
    # copybara:uncomment_begin(google-only)
    # visibility = ["//tensorflow:internal"],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/eager/polymorphic_function:eager_function_run",
    ],
)

py_strict_library(
    name = "lift_to_graph",
    srcs = ["lift_to_graph.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:op_selector",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "lift_to_graph_test",
    size = "medium",
    srcs = ["lift_to_graph_test.py"],
    python_version = "PY3",
    deps = [
        "lift_to_graph",
        ":def_function",
        ":test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_library(
    name = "wrap_function",
    srcs = ["wrap_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":function",
        ":lift_to_graph",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/python/eager/polymorphic_function:atomic_function",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "wrap_function_test",
    srcs = ["wrap_function_test.py"],
    python_version = "PY3",
    deps = [
        ":backprop",
        ":def_function",
        ":wrap_function",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:saver",
    ],
)

cuda_py_strict_test(
    name = "wrap_function_device_test",
    srcs = ["wrap_function_device_test.py"],
    python_version = "PY3",
    xla_enable_strict_auto_jit = False,
    xla_enabled = False,
    deps = [
        ":def_function",
        ":wrap_function",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "remote",
    srcs = ["remote.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:remote_utils",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ] + tf_additional_rpc_deps(),
)

cuda_py_strict_test(
    name = "remote_test",
    size = "medium",
    srcs = ["remote_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "no_oss",  # This test launches local server.
        "nofastbuild",  # times out
        "optonly",  # times out
    ],
    deps = [
        ":cancellation",
        ":context",
        ":def_function",
        ":executor",
        ":remote",
        ":test",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:test_ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:functional_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/testing:parameterized",
        "@pypi_portpicker//:pkg",
    ],
)

cuda_py_strict_test(
    name = "remote_execution_test",
    srcs = ["remote_execution_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_oss",  # This test launches local server
    ],
    deps = [
        ":backprop",
        ":context",
        ":def_function",
        ":remote",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:script_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
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
        ":context",
        ":def_function",
        ":executor",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/distribute/failure_handling:check_preemption_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:coordinator",
        "//tensorflow/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "remote_cloud_tpu_test",
    srcs = ["remote_cloud_tpu_test.py"],
    python_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":remote",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/framework:config",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
)

tpu_py_strict_test(
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
        ":remote",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/tpu:tpu_strategy_util",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
)

cuda_py_strict_test(
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
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "gradient_input_output_exclusions",
    srcs = ["gradient_input_output_exclusions.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:cfg",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:transformer",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:liveness",
        "//tensorflow/python/autograph/pyct/static_analysis:reaching_fndefs",
        "//tensorflow/python/framework:op_def_registry",
        "//tensorflow/python/framework:ops",
        "@pypi_gast//:pkg",
    ],
)

py_strict_binary(
    name = "gen_gradient_input_output_exclusions",
    srcs = ["gen_gradient_input_output_exclusions.py"],
    deps = [":gradient_input_output_exclusions"],
)

# Needed for the test below.
exports_files([
    "pywrap_gradient_exclusions.cc",
])

py_strict_test(
    name = "gradient_input_output_exclusions_test",
    srcs = ["gradient_input_output_exclusions_test.py"],
    data = [
        ":pywrap_gradient_exclusions.cc",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",  # No point linking the gen script in the pip package.
    ],
    deps = [
        ":gradient_input_output_exclusions",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:resource_loader",
    ],
)
