# Description:
# TensorFlow SavedModel.

load("//tensorflow:tensorflow.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_pybind_cc_library_wrapper")

package(
    # TODO(drpng): change that to //third_party/tensorflow:internal
    # when we have migrated all users.
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "saved_model",
    srcs = ["saved_model.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":builder",
        ":constants",
        ":load",
        ":loader",
        ":main_op",
        ":method_name_updater",
        ":save",
        ":signature_constants",
        ":signature_def_utils",
        ":simple_save",
        ":tag_constants",
        ":utils",
        "//tensorflow/python:util",
        "//tensorflow/python/saved_model/model_utils",
        "//tensorflow/python/saved_model/registration",
    ],
)

py_strict_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY3",
    deps = [
        ":pywrap_saved_model",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "signature_constants",
    srcs = ["signature_constants.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python/util:tf_export"],
)

py_strict_library(
    name = "tag_constants",
    srcs = ["tag_constants.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python/util:tf_export"],
)

py_strict_library(
    name = "builder",
    srcs = [
        "builder.py",
        "builder_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":pywrap_saved_model",
        ":signature_def_utils",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:saver",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "loader",
    srcs = [
        "loader.py",
        "loader_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":pywrap_saved_model",
        ":signature_def_utils",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:saver",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "loader_test",
    size = "small",
    srcs = ["loader_test.py"],
    deps = [
        ":builder",
        ":loader",
        ":signature_def_utils",
        ":utils",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "simple_save",
    srcs = [
        "simple_save.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":builder",
        ":signature_constants",
        ":signature_def_utils",
        ":tag_constants",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "main_op",
    srcs = [
        "main_op.py",
        "main_op_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "saved_model_test",
    size = "small",
    srcs = ["saved_model_test.py"],
    data = ["//tensorflow/cc/saved_model:saved_model_half_plus_two"],
    tags = ["no_windows"],
    deps = [
        ":builder",
        ":constants",
        ":loader",
        ":main_op",
        ":signature_def_utils",
        ":tag_constants",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:test_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/training:saver_test_utils",
    ],
)

py_strict_library(
    name = "utils",
    srcs = [
        "utils.py",
        "utils_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":nested_structure_coder",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:lib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_tensor",
    ],
)

py_strict_library(
    name = "signature_def_utils",
    srcs = [
        "signature_def_utils.py",
        "signature_def_utils_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":signature_constants",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "signature_def_utils_test",
    size = "small",
    srcs = ["signature_def_utils_test.py"],
    deps = [
        ":signature_constants",
        ":signature_def_utils",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "simple_save_test",
    size = "small",
    srcs = ["simple_save_test.py"],
    deps = [
        ":loader",
        ":signature_constants",
        ":simple_save",
        ":tag_constants",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:variables",
    ],
)

py_strict_library(
    name = "signature_serialization",
    srcs = [
        "signature_serialization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":function_serialization",
        ":revived_types",
        ":signature_constants",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "save_context",
    srcs = [
        "save_context.py",
    ],
    srcs_version = "PY3",
)

tf_py_test(
    name = "save_context_test",
    srcs = ["save_context_test.py"],
    srcs_version = "PY3",
    deps = [
        ":save_context",
        ":save_options",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python/eager:test",
    ],
)

py_strict_library(
    name = "save",
    srcs = [
        "save.py",
        "tracing_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":builder",
        ":function_serialization",
        ":pywrap_saved_model",
        ":revived_types",
        ":save_context",
        ":save_options",
        ":signature_constants",
        ":signature_def_utils",
        ":signature_serialization",
        ":tag_constants",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:error_interpolation",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:lib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python:versions",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/checkpoint:functional_saver",
        "//tensorflow/python/checkpoint:graph_view",
        "//tensorflow/python/checkpoint:saveable_compat",
        "//tensorflow/python/checkpoint:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:function_saved_model_utils",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:resource",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

tf_py_test(
    name = "save_test",
    srcs = ["save_test.py"],
    deps = [
        ":loader",
        ":save",
        ":save_options",
        ":signature_constants",
        ":tag_constants",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:error_interpolation",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "load",
    srcs = [
        "load.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":function_deserialization",
        ":load_options",
        ":load_v1_in_v2",
        ":loader",
        ":pywrap_saved_model",
        ":revived_types",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/checkpoint:graph_view",
        "//tensorflow/python/checkpoint:restore",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:distribute_utils",
        "//tensorflow/python/distribute:values_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:function_saved_model_utils",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/python/trackable:resource",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "load_v1_in_v2",
    srcs = [
        "load_v1_in_v2.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":function_deserialization",
        ":loader",
        ":pywrap_saved_model",
        ":signature_serialization",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:platform",
        "//tensorflow/python:saver",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:lift_to_graph",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:resource",
        "//tensorflow/python/training:monitored_session",
    ],
)

cuda_py_test(
    name = "load_test",
    srcs = ["load_test.py"],
    shard_count = 10,
    tags = [
        "no_gpu",  # TODO(b/136560979): flaky
        "no_mac",  # TODO(b/124822121): Re-enable this test.
    ],
    deps = [
        ":load",
        ":save",
        "//tensorflow/python:cond_v2",  # b/118513001
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:while_v2",  # b/118513001
        "//tensorflow/python/checkpoint:saveable_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/module",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:resource",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "load_v1_in_v2_test",
    srcs = ["load_v1_in_v2_test.py"],
    deps = [
        ":builder",
        ":load",
        ":save",
        ":signature_def_utils",
        ":simple_save",
        ":utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "revived_types",
    srcs = [
        "revived_types.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "revived_types_test",
    srcs = ["revived_types_test.py"],
    deps = [
        ":revived_types",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
    ],
)

py_strict_library(
    name = "function_serialization",
    srcs = [
        "function_serialization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":nested_structure_coder",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:function",
    ],
)

py_strict_library(
    name = "function_deserialization",
    srcs = [
        "function_deserialization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":nested_structure_coder",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:default_gradient",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:function_def_to_graph",
        "//tensorflow/python:gradients",
        "//tensorflow/python:op_def_registry",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tf_decorator",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:function_spec",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "nested_structure_coder",
    srcs = ["nested_structure_coder.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:indexed_slices",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:optional_ops",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:row_partition",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "nested_structure_coder_test",
    srcs = ["nested_structure_coder_test.py"],
    deps = [
        ":nested_structure_coder",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python/eager:test",
    ],
)

py_strict_library(
    name = "save_options",
    srcs = ["save_options.py"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "load_options",
    srcs = ["load_options.py"],
    deps = [
        ":save_options",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "method_name_updater",
    srcs = ["method_name_updater.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":loader",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "method_name_updater_test",
    srcs = ["method_name_updater_test.py"],
    deps = [
        ":method_name_updater",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python/eager:test",
    ],
)

tf_py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":pywrap_saved_model",
        "//tensorflow/python/eager:test",
    ],
)

tf_pybind_cc_library_wrapper(
    name = "pywrap_saved_model_headers",
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/cc/saved_model:fingerprinting",
        "//tensorflow/cc/saved_model:metrics",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_saved_model",
    srcs = [
        "pywrap_saved_model.cc",
        "pywrap_saved_model_constants.cc",
        "pywrap_saved_model_fingerprinting.cc",
        "pywrap_saved_model_metrics.cc",
    ],
    hdrs = [
        "pywrap_saved_model_constants.h",
        "pywrap_saved_model_fingerprinting.h",
        "pywrap_saved_model_metrics.h",
    ],
    features = ["-layering_check"],
    visibility = ["//tensorflow/python/training:__subpackages__"],
    deps = [
        ":pywrap_saved_model_headers",
        "//tensorflow/cc/experimental/libexport:save",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

tf_py_test(
    name = "pywrap_saved_model_metrics_test",
    srcs = ["pywrap_saved_model_metrics_test.py"],
    deps = [
        ":pywrap_saved_model",
        "//tensorflow/python/eager:test",
    ],
)

tf_py_test(
    name = "keras_injection_test",
    size = "small",
    srcs = ["keras_injection_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "pywrap_saved_model_fingerprinting_test",
    srcs = ["pywrap_saved_model_fingerprinting_test.py"],
    data = ["//tensorflow/cc/saved_model:saved_model_test_files"],
    tags = ["no_windows"],
    deps = [
        ":pywrap_saved_model",
        "//tensorflow/python/eager:test",
    ],
)

tf_py_test(
    name = "fingerprinting_test",
    size = "small",
    srcs = ["fingerprinting_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python/eager:test",
    ],
)

tf_py_test(
    name = "tracing_utils_test",
    size = "small",
    srcs = ["tracing_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":save",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/trackable:base",
    ],
)
