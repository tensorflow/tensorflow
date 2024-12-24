# Description:
# TensorFlow SavedModel.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test", "tf_pybind_cc_library_wrapper", "tf_python_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        ":fingerprinting",
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
        ":fingerprinting_utils",
        ":path_helpers",
        ":pywrap_saved_model",
        ":signature_def_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        # copybara:uncomment "//tensorflow/tools/proto_splitter/python:saved_model",
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
        ":path_helpers",
        ":pywrap_saved_model",
        ":signature_def_utils",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "loader_test",
    size = "small",
    srcs = ["loader_test.py"],
    deps = [
        ":builder",
        ":loader",
        ":signature_def_utils",
        ":utils",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:resource_variables_toggle",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:saver",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "simple_save",
    srcs = ["simple_save.py"],
    srcs_version = "PY3",
    deps = [
        ":builder",
        ":signature_constants",
        ":signature_def_utils",
        ":tag_constants",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
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
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "saved_model_test",
    size = "small",
    srcs = ["saved_model_test.py"],
    data = ["//tensorflow/cc/saved_model:saved_model_half_plus_two"],
    tags = ["no_windows"],
    deps = [
        ":builder",
        ":constants",
        ":fingerprinting",
        ":loader",
        ":main_op",
        ":signature_def_utils",
        ":tag_constants",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:test_ops",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variables_toggle",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training",
        "//tensorflow/python/training:saver_test_utils",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_library(
    name = "path_helpers",
    srcs = ["path_helpers.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/util:compat",
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
        ":nested_structure_coder",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:byte_swap_tensor",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    deps = [
        ":nested_structure_coder",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
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
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "signature_def_utils_test",
    size = "small",
    srcs = ["signature_def_utils_test.py"],
    deps = [
        ":signature_constants",
        ":signature_def_utils",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "simple_save_test",
    size = "small",
    srcs = ["simple_save_test.py"],
    deps = [
        ":loader",
        ":signature_constants",
        ":simple_save",
        ":tag_constants",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "signature_serialization",
    srcs = ["signature_serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":function_serialization",
        ":revived_types",
        ":signature_constants",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager/polymorphic_function:attributes",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "save_context",
    srcs = ["save_context.py"],
    srcs_version = "PY3",
)

tf_py_strict_test(
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
    srcs = ["save.py"],
    srcs_version = "PY3",
    deps = [
        ":builder",
        ":fingerprinting_utils",
        ":function_serialization",
        ":path_helpers",
        ":pywrap_saved_model",
        ":revived_types",
        ":save_context",
        ":save_options",
        ":signature_constants",
        ":signature_def_utils",
        ":signature_serialization",
        ":tag_constants",
        ":tracing_utils",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/checkpoint:functional_saver",
        "//tensorflow/python/checkpoint:graph_view",
        "//tensorflow/python/checkpoint:save_util_v1",
        "//tensorflow/python/checkpoint:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/eager/polymorphic_function:concrete_function",
        "//tensorflow/python/eager/polymorphic_function:saved_model_exported_concrete",
        "//tensorflow/python/eager/polymorphic_function:saved_model_utils",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:error_interpolation",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:versions",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:resource",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/training/saving:trace_saveable_util",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_stack",
        "@absl_py//absl/logging",
    ] + if_google([
        "//tensorflow/tools/proto_splitter/python:saved_model",
    ]),
)

py_strict_library(
    name = "tracing_utils",
    srcs = [
        "tracing_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/checkpoint:saveable_compat",
        "//tensorflow/python/checkpoint:tensor_callable",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
    ],
)

tf_py_strict_test(
    name = "save_test",
    srcs = ["save_test.py"],
    deps = [
        ":load",
        ":loader",
        ":save",
        ":save_options",
        ":signature_constants",
        ":tag_constants",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint/sharding:sharding_policies",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:versions",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_switch_case",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

# copybara:uncomment_begin(google-only)
#
# tf_py_strict_test(
#     name = "proto_splitter_save_test",
#     srcs = ["proto_splitter_save_test.py"],
#     tags = [
#         "no_mac",  # TODO(b/291933687)
#         "no_windows",  # TODO(b/291001524)
#     ],
#     deps = [
#         ":save",
#         ":save_options",
#         "//tensorflow/python/eager:def_function",
#         "//tensorflow/python/eager:test",
#         "//tensorflow/python/framework:constant_op",
#         "//tensorflow/python/module",
#         "//tensorflow/tools/proto_splitter:constants",
#         "//third_party/py/numpy",
#         "@absl_py//absl/testing:parameterized",
#     ],
# )
#
# copybara:uncomment_end

py_strict_library(
    name = "load",
    srcs = ["load.py"],
    srcs_version = "PY3",
    tags = [
        "ignore_for_dep=third_party.py.tf_keras.optimizers.optimizer_v2",
    ],
    deps = [
        ":fingerprinting",
        ":fingerprinting_utils",
        ":function_deserialization",
        ":load_options",
        ":load_v1_in_v2",
        ":loader",
        ":path_helpers",
        ":pywrap_saved_model",
        ":revived_types",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/capture:restore_captures",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/checkpoint:graph_view",
        "//tensorflow/python/checkpoint:restore",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:distribute_utils",
        "//tensorflow/python/distribute:values_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager/polymorphic_function:saved_model_utils",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/python/trackable:resource",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/training:py_checkpoint_reader",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "load_v1_in_v2",
    srcs = ["load_v1_in_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":function_deserialization",
        ":loader",
        ":pywrap_saved_model",
        ":signature_serialization",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:lift_to_graph",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:resource",
        "//tensorflow/python/training:monitored_session",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/util:nest",
    ],
)

cuda_py_strict_test(
    name = "load_test",
    srcs = ["load_test.py"],
    shard_count = 10,
    tags = [
        "no_gpu",  # TODO(b/136560979): flaky
        "no_mac",  # TODO(b/124822121): Re-enable this test.
    ],
    deps = [
        ":load",
        ":load_options",
        ":loader",
        ":save",
        ":save_options",
        ":tag_constants",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:saveable_compat",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:op_callbacks",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:versions",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond_v2",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:resource",
        "//tensorflow/python/training:monitored_session",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:tf_inspect",
        "@absl_py//absl/testing:parameterized",
    ] + if_google([
        "//tensorflow/cc/experimental/tf2:runtime_pybind",
    ]),
)

tf_py_strict_test(
    name = "load_v1_in_v2_test",
    srcs = ["load_v1_in_v2_test.py"],
    deps = [
        ":builder",
        ":load",
        ":save",
        ":signature_def_utils",
        ":simple_save",
        ":tag_constants",
        ":utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:lift_to_graph",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:versions",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:ref_variable",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/training:saver",
    ],
)

tf_py_strict_test(
    name = "load_optimizer_test",
    srcs = ["load_optimizer_test.py"],
    data = ["//tensorflow/cc/saved_model:saved_model_test_files"],
    tags = ["no_oss"],  # Due to the usage of keras component.
    deps = [
        ":load",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/ops:variables",
    ],
)

py_strict_library(
    name = "revived_types",
    srcs = ["revived_types.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "revived_types_test",
    srcs = ["revived_types_test.py"],
    deps = [
        ":revived_types",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/trackable:autotrackable",
    ],
)

py_strict_library(
    name = "function_serialization",
    srcs = ["function_serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":nested_structure_coder",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/eager/polymorphic_function:function_type_utils",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "function_deserialization",
    srcs = ["function_deserialization.py"],
    srcs_version = "PY3",
    deps = [
        ":nested_structure_coder",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager/polymorphic_function:function_type_utils",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function_def_to_graph",
        "//tensorflow/python/framework:op_def_registry",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:default_gradient",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_inspect",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "nested_structure_coder",
    srcs = ["nested_structure_coder.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:type_spec_registry",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "nested_structure_coder_test",
    srcs = ["nested_structure_coder_test.py"],
    deps = [
        ":nested_structure_coder",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:immutable_dict",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/framework:type_spec_registry",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types:internal",
    ],
)

py_strict_library(
    name = "save_options",
    srcs = ["save_options.py"],
    deps = [
        "//tensorflow/python/checkpoint/sharding:sharding_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
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
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "method_name_updater_test",
    srcs = ["method_name_updater_test.py"],
    deps = [
        ":constants",
        ":loader",
        ":method_name_updater",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":builder",
        ":fingerprinting",
        ":load",
        ":load_v1_in_v2",
        ":loader",
        ":pywrap_saved_model",
        ":save",
        "//tensorflow/python/checkpoint/sharding:sharding_policies",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/trackable:autotrackable",
    ],
)

tf_pybind_cc_library_wrapper(
    name = "pywrap_saved_model_headers",
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/cc/saved_model:fingerprinting",
        "//tensorflow/cc/saved_model:metrics",
    ] + if_google([
        "//tensorflow/tools/proto_splitter:merge",
    ]),
)

tf_python_pybind_extension(
    name = "pywrap_saved_model",
    srcs = [
        "pywrap_saved_model.cc",
        "pywrap_saved_model_constants.cc",
        "pywrap_saved_model_fingerprinting.cc",
        "pywrap_saved_model_metrics.cc",
    ] + if_google([
        "pywrap_saved_model_merger.cc",
    ]),
    hdrs = [
        "pywrap_saved_model_constants.h",
        "pywrap_saved_model_fingerprinting.h",
        "pywrap_saved_model_metrics.h",
    ] + if_google([
        "pywrap_saved_model_merger.h",
    ]),
    enable_stub_generation = True,
    # This fails Windows builds. Please check b/266870200 for details.
    #    dynamic_deps = ["//tensorflow/python:_pywrap_tensorflow_internal.so"] + select({
    #        "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.%s.dylib" % VERSION],
    #        "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION],
    #        "//tensorflow:windows": [],
    #    }),
    #    static_deps = tf_python_pybind_static_deps(),
    pytype_srcs = [
        "pywrap_saved_model/__init__.pyi",
        "pywrap_saved_model/constants.pyi",
        "pywrap_saved_model/fingerprinting.pyi",
        "pywrap_saved_model/merger.pyi",
        "pywrap_saved_model/metrics.pyi",
    ],
    visibility = [
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/checkpoint:__subpackages__",
        "//tensorflow/python/tpu:__pkg__",
        "//tensorflow/python/training:__subpackages__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    deps = [
        # placeholder for index annotation deps
        "//tensorflow/cc/experimental/libexport:save",
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/cc/saved_model:fingerprinting",
        "//tensorflow/cc/saved_model:metrics",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:path",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_google([
        "//tensorflow/tools/proto_splitter:merge",
    ]),
)

tf_py_strict_test(
    name = "pywrap_saved_model_metrics_test",
    srcs = ["pywrap_saved_model_metrics_test.py"],
    deps = [
        ":pywrap_saved_model",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:test",
    ],
)

# copybara:uncomment_begin(google-only)
#
# tf_py_strict_test(
#     name = "pywrap_saved_model_merger_test",
#     srcs = ["pywrap_saved_model_merger_test.py"],
#     data = [
#         "//tensorflow/tools/proto_splitter/testdata:split-standard.cpb",
#         "//tensorflow/tools/proto_splitter/testdata:split-standard.pbtxt",
#     ],
#     deps = [
#         ":pywrap_saved_model",
#         "//tensorflow/core:protos_all_py",
#         "//tensorflow/python/eager:test",
#         "//tensorflow/python/lib/io:file_io",
#         "//tensorflow/python/platform:client_testlib",
#     ],
# )
#
# copybara:uncomment_end

tf_py_strict_test(
    name = "keras_injection_test",
    size = "small",
    srcs = ["keras_injection_test.py"],
    data = ["//tensorflow/cc/saved_model:saved_model_test_files"],
    python_version = "PY3",
    tags = ["no_oss"],  # Due to the usage of keras component.
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/eager:test",
    ],
)

tf_py_strict_test(
    name = "pywrap_saved_model_fingerprinting_test",
    srcs = ["pywrap_saved_model_fingerprinting_test.py"],
    data = ["//tensorflow/cc/saved_model:saved_model_test_files"],
    tags = ["no_windows"],
    deps = [
        ":pywrap_saved_model",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "fingerprinting",
    srcs = ["fingerprinting.py"],
    deps = [
        ":pywrap_saved_model",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "fingerprinting_utils",
    srcs = ["fingerprinting_utils.py"],
    deps = [
        ":fingerprinting",
        ":pywrap_saved_model",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "fingerprinting_test",
    size = "small",
    srcs = ["fingerprinting_test.py"],
    python_version = "PY3",
    deps = [
        ":fingerprinting",
        ":fingerprinting_utils",
        ":pywrap_saved_model",
        ":save",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/trackable:autotrackable",
    ],
)

tf_py_strict_test(
    name = "tracing_utils_test",
    size = "small",
    srcs = ["tracing_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":tracing_utils",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/trackable:base",
    ],
)

# copybara:uncomment_begin(google-only)
#
# tf_cc_test(
#     name = "lineage_logging_test",
#     srcs = ["lineage_logging_test.cc"],
#     deps = [
#         "//learning/brain/contrib/hub/public:client",
#         "//learning/brain/contrib/hub/public:handle",
#         "//learning/brain/contrib/hub/public/proto:metadata_cc_proto",
#         "//learning/brain/contrib/hub/public/proto:options_cc_proto",
#         "//learning/brain/contrib/hub/public/testing:in_process_tfhub_env",
#         "//learning/brain/contrib/hub/public/testing:testdata_util",
#         "//learning/metadata/cataline/proto:ml_data_processes_cc_proto",
#         "//learning/metadata/lineage_log/cc:lineage_log",
#         "//learning/metadata/lineage_log/cc:mock_process_client",
#         "//learning/metadata/lineage_log/cc:test_util",
#         "//tensorflow/core/platform:protobuf",
#         "//tensorflow/core/protobuf:for_core_protos_cc",
#         "@local_tsl//tsl/platform:env",
#         "@local_tsl//tsl/platform:path",
#         "//third_party/protobuf",
#         "//third_party/protobuf/io",
#         "@com_google_absl//absl/log:check",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_googletest//:gtest_main",
#     ],
# )
#
# copybara:uncomment_end
