load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "attributes",
    srcs = ["attributes.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python:__subpackages__"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_library(
    name = "autograph_util",
    srcs = ["autograph_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/impl:api",
        "//tensorflow/python/util:tf_decorator_py",
    ],
)

py_strict_library(
    name = "transform",
    srcs = ["transform.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python:__subpackages__"],
    deps = [],
)

pytype_strict_library(
    name = "atomic_function",
    srcs = ["atomic_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python:__subpackages__"],
    deps = [
        ":attributes",
        ":function_type_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:auto_control_deps_utils",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:error_interpolation",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function_def_to_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:handle_data_util",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:function_utils",
        "//tensorflow/python/util:tf_stack",
    ],
)

cuda_py_strict_test(
    name = "atomic_function_test",
    size = "medium",
    srcs = ["atomic_function_test.py"],
    python_version = "PY3",
    deps = [
        ":atomic_function",
        ":polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "concrete_function",
    srcs = ["concrete_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python:__subpackages__"],
    deps = [
        ":atomic_function",
        ":attributes",
        ":function_type_utils",
        ":saved_model_exported_concrete",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/eager:backprop_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:forwardprop_util",
        "//tensorflow/python/eager:graph_only_ops",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:default_gradient",
        "//tensorflow/python/ops:gradients_util",
        "//tensorflow/python/ops:handle_data_util",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
    ],
)

cuda_py_strict_test(
    name = "concrete_function_test",
    size = "small",
    srcs = ["concrete_function_test.py"],
    python_version = "PY3",
    deps = [
        ":atomic_function",
        ":concrete_function",
        ":polymorphic_function",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "tracing_compilation",
    srcs = ["tracing_compilation.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python:__subpackages__"],
    deps = [
        ":attributes",
        ":concrete_function",
        ":function_context",
        ":function_type_utils",
        ":tf_method_target",
        ":transform",
        "//tensorflow/core/function/capture:capture_container",
        "//tensorflow/core/function/polymorphism:function_cache",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_library(
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

py_strict_library(
    name = "polymorphic_function",
    srcs = ["polymorphic_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":attributes",
        ":autograph_util",
        ":compiler_ir",
        ":eager_function_run",
        ":function_type_utils",
        ":tf_method_target",
        ":tracing_compilation",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/capture:capture_container",
        "//tensorflow/core/function/polymorphism:function_cache",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/distribute/parallel_device",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:lift_to_graph",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:traceback_utils",
    ],
)

cuda_py_strict_test(
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
        ":attributes",
        ":polymorphic_function",
        "//tensorflow/core/function/capture:capture_container",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:cancellation",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:lift_to_graph",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:test_ops",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:clip_ops",
        "//tensorflow/python/ops:cond_v2",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:functional_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:random_ops_gen",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:script_ops",
        "//tensorflow/python/ops:sendrecv_ops_gen",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/structured:structured_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
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
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "polymorphic_function_xla_jit_test",
    srcs = ["polymorphic_function_xla_jit_test.py"],
    # copybara:uncomment_begin
    # #TODO(b/185944215) # Remove once the bug is fixed.
    # disable_tpu_tfrt = True,
    # copybara:uncomment_end
    disabled_backends = [
        "cpu_ondemand",
    ],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_mac",
        "no_oss",  # TODO(b/295654746)
        "no_pip",
        "no_tfrt",  # TODO(b/185944215)
        "no_windows",
    ],
    use_xla_device = False,
    deps = [
        ":polymorphic_function",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:collective_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
    ],
)

tf_xla_py_strict_test(
    name = "polymorphic_function_xla_test",
    srcs = ["polymorphic_function_xla_test.py"],
    # copybara:uncomment_begin
    # #TODO(b/185944215) # Remove once the bug is fixed.
    # disable_tpu_tfrt = True,
    # copybara:uncomment_end
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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "gradients_test",
    size = "medium",
    srcs = ["gradients_test.py"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":polymorphic_function",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
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

tf_py_strict_test(
    name = "tracing_compilation_test",
    size = "medium",
    srcs = ["tracing_compilation_test.py"],
    python_version = "PY3",
    deps = [
        ":function_type_utils",
        ":tracing_compilation",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/capture:capture_container",
        "//tensorflow/core/function/polymorphism:function_cache",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:test_ops",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:functional_ops_gen",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:resource_variable_ops_gen",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "argument_naming_test",
    size = "medium",
    srcs = ["argument_naming_test.py"],
    python_version = "PY3",
    # TODO(b/169371527): insert transfer op in eager lowering for TFRT.
    deps = [
        ":polymorphic_function",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "collection_test",
    size = "medium",
    srcs = ["collection_test.py"],
    python_version = "PY3",
    deps = [
        ":polymorphic_function",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "eager_function_run",
    srcs = ["eager_function_run.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "function_context",
    srcs = ["function_context.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core/function/polymorphism:function_cache",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/saved_model:save_context",
    ],
)

py_strict_library(
    name = "saved_model_utils",
    srcs = ["saved_model_utils.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:base",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "saved_model_exported_concrete",
    srcs = ["saved_model_exported_concrete.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":function_type_utils",
        "//tensorflow/python/trackable:base",
    ],
)

py_strict_library(
    name = "function_type_utils",
    srcs = ["function_type_utils.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":composite_tensor_utils",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_py_strict_test(
    name = "function_spec_test",
    size = "medium",
    srcs = ["function_spec_test.py"],
    python_version = "PY3",
    deps = [
        ":function_type_utils",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:tf_decorator_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "compiler_ir",
    srcs = ["compiler_ir.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/util:nest",
    ],
)

tf_xla_py_strict_test(
    name = "compiler_ir_test",
    srcs = ["compiler_ir_test.py"],
    disabled_backends = [
        "cpu_ondemand",
        "gpu_a100",
        "gpu_h100",
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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
    ],
)
