load("//tensorflow:tensorflow.default.bzl", "cuda_py_test", "tf_py_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "attributes",
    srcs = ["attributes.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "atomic_function",
    srcs = ["atomic_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python:__subpackages__"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:handle_data_util",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/eager:cancellation",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/framework:c_api_util",
        "//tensorflow/python/framework:error_interpolation",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:function_utils",
    ],
)

py_library(
    name = "monomorphic_function",
    srcs = ["monomorphic_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python:__subpackages__"],
    deps = [
        ":atomic_function",
        ":attributes",
        ":function_spec",
        ":saved_model_exported_concrete",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:default_gradient",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:handle_data_util",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:backprop_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:forwardprop_util",
        "//tensorflow/python/eager:graph_only_ops",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
    ],
)

py_library(
    name = "tracing_compiler",
    srcs = ["tracing_compiler.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python:__subpackages__"],
    deps = [
        ":attributes",
        ":function_context",
        ":monomorphic_function",
        ":tf_method_target",
        "//tensorflow/core/function/capture:capture_container",
        "//tensorflow/core/function/polymorphism:function_cache",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:lazy_loader",
        "//tensorflow/python/util:tf_decorator",
    ],
)

py_library(
    name = "tf_method_target",
    srcs = ["tf_method_target.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow/python/autograph/impl:__pkg__",
        "//tensorflow/python/eager:__pkg__",
    ],
    deps = [
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_library(
    name = "polymorphic_function",
    srcs = ["polymorphic_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":attributes",
        ":compiler_ir",
        ":eager_function_run",
        ":function_spec",
        "//tensorflow/python:cond",
        "//tensorflow/python:cond_v2",  # TODO(b/118513001): Imported via control_flow_ops; remove.
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:while_v2",  # TODO(b/118513001): Imported via control_flow_ops; remove.
        "//tensorflow/python/distribute/parallel_device",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:lift_to_graph",
        "//tensorflow/python/eager/polymorphic_function:monomorphic_function",
        "//tensorflow/python/eager/polymorphic_function:tracing_compiler",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:traceback_utils",
    ],
)

cuda_py_test(
    name = "polymorphic_function_test",
    size = "medium",
    srcs = ["polymorphic_function_test.py"],
    python_version = "PY3",
    shard_count = 15,
    tags = [
        "nomac",  # b/157056289
    ],
    # TODO(b/169371527): insert transfer op in eager lowering for TFRT.
    deps = [
        ":monomorphic_function",
        ":polymorphic_function",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_assert",
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
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:cancellation",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "polymorphic_function_test_cpu_only",
    srcs = ["polymorphic_function_test_cpu_only.py"],
    python_version = "PY3",
    # --config=cuda implicitly links in XLA.
    tags = [
        "no_cuda_on_cpu_tap",
        "no_gpu",
        "no_oss",  # No way to force no XLA linkage in OSS build from here.
        "no_pip",
    ],
    deps = [
        ":polymorphic_function",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "polymorphic_function_xla_jit_test",
    srcs = ["polymorphic_function_xla_jit_test.py"],
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
        ":polymorphic_function",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/util:nest",
    ],
)

tf_xla_py_test(
    name = "polymorphic_function_xla_test",
    srcs = ["polymorphic_function_xla_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":polymorphic_function",
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

cuda_py_test(
    name = "gradients_test",
    size = "medium",
    srcs = ["gradients_test.py"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":polymorphic_function",
        "//tensorflow/python:cond",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "quarantine",
    srcs = ["quarantine.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/eager:__pkg__"],
    deps = [
        ":atomic_function",
        ":eager_function_run",
        ":polymorphic_function",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "composite_tensor_utils",
    srcs = ["composite_tensor_utils.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:nest",
    ],
)

tf_py_test(
    name = "quarantine_test",
    size = "medium",
    srcs = ["quarantine_test.py"],
    python_version = "PY3",
    deps = [
        ":monomorphic_function",
        ":polymorphic_function",
        ":quarantine",
        "//tensorflow/python:client_testlib",
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
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:cancellation",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "argument_naming_test",
    size = "medium",
    srcs = ["argument_naming_test.py"],
    python_version = "PY3",
    # TODO(b/169371527): insert transfer op in eager lowering for TFRT.
    deps = [
        ":polymorphic_function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "collection_test",
    size = "medium",
    srcs = ["collection_test.py"],
    python_version = "PY3",
    deps = [
        ":polymorphic_function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "eager_function_run",
    srcs = ["eager_function_run.py"],
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/python/util:tf_export"],
)

py_library(
    name = "function_context",
    srcs = ["function_context.py"],
    srcs_version = "PY3",
    visibility = [
        "//visibility:private",  # Only private by automation, not intent. Owner may accept CLs adding visibility. See go/scheuklappen#explicit-private.
    ],
    deps = [
        "//tensorflow/core/function/polymorphism:function_cache",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/saved_model:save_context",
    ],
)

py_library(
    name = "saved_model_utils",
    srcs = ["saved_model_utils.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:base",
    ],
)

py_library(
    name = "saved_model_exported_concrete",
    srcs = ["saved_model_exported_concrete.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/trackable:base",
    ],
)

py_library(
    name = "function_spec",
    srcs = ["function_spec.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":composite_tensor_utils",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "function_spec_test",
    size = "medium",
    srcs = ["function_spec_test.py"],
    python_version = "PY3",
    deps = [
        ":function_spec",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/util:tf_decorator",
    ],
)

py_library(
    name = "compiler_ir",
    srcs = ["compiler_ir.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/util:nest",
    ],
)

tf_xla_py_test(
    name = "compiler_ir_test",
    srcs = ["compiler_ir_test.py"],
    disabled_backends = [
        "cpu_ondemand",
        "gpu_a100",
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
        ":compiler_ir",
        ":polymorphic_function",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/util:nest",
    ],
)
