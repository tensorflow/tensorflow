load("//tensorflow:tensorflow.default.bzl", "cuda_py_test", "tf_py_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "monomorphic_function",
    srcs = ["monomorphic_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/eager:__pkg__"],
    deps = [
        ":function_context",
        ":function_spec",
        ":saved_model_utils",
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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:forwardprop_util",
        "//tensorflow/python/eager:graph_only_ops",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tracing_compiler",
    srcs = ["tracing_compiler.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/eager:__pkg__"],
    deps = [
        ":monomorphic_function",
    ],
)

py_library(
    name = "polymorphic_function",
    srcs = ["polymorphic_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":function_spec",
        "//tensorflow/python:cond_v2",  # TODO(b/118513001): Imported via control_flow_ops; remove.
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:while_v2",  # TODO(b/118513001): Imported via control_flow_ops; remove.
        "//tensorflow/python/distribute/parallel_device",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:lift_to_graph",
        "//tensorflow/python/eager/polymorphic_function:monomorphic_function",
        "//tensorflow/python/eager/polymorphic_function:tracing_compiler",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/trackable:base",
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
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:cancellation",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/eager/polymorphic_function:monomorphic_function",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model:save_options",
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
        "//tensorflow/python/autograph/core",
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
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:backprop",
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
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "quarantine",
    srcs = ["quarantine.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/eager:__pkg__"],
    deps = [
        ":monomorphic_function",
        ":tracing_compiler",
    ],
)

py_library(
    name = "composite_tensor_utils",
    srcs = ["composite_tensor_utils.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/util",
    ],
)

tf_py_test(
    name = "quarantine_test",
    size = "medium",
    srcs = ["quarantine_test.py"],
    python_version = "PY3",
    deps = [
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
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/eager/polymorphic_function:monomorphic_function",
        "//tensorflow/python/eager/polymorphic_function:quarantine",
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
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "function_context",
    srcs = ["function_context.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
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
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:resource",
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
        "//tensorflow/python/util",
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
    ],
)
