# Description: Operations defined for Cloud TPUs
load("//tensorflow:pytype.default.bzl", "pytype_library", "pytype_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_test")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")

# Do not add anymore paths here. You do not need to be in the visibility list
# to use TPU symbols. They are accessible from tf.contrib.tpu in TF 1.x and
# tf.tpu and tf.compat.v1.tpu in TF 2.x.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//learning/brain:__subpackages__",
        "//learning/deepmind:__subpackages__",
        "//learning/serving:__subpackages__",
        "//research/graph:__subpackages__",
        "//tensorflow:__subpackages__",
        "//third_party/py/jax_tpu_embedding:__subpackages__",
        "//third_party/py/lingvo:__subpackages__",
        "//waymo/ml/deploy/sync_test/tools:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["tpu_test_wrapper.py"])

py_test(
    name = "tpu_test_wrapper_test",
    srcs = [
        "tpu_test_wrapper.py",
        "tpu_test_wrapper_test.py",
    ],
    main = "tpu_test_wrapper_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss_py35",
        "no_pip",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
        "@absl_py//absl/testing:flagsaver",
    ],
)

alias(
    name = "tpu_ops",
    actual = "//tensorflow/python/tpu/ops",
)

pytype_library(
    name = "async_checkpoint",
    srcs = ["async_checkpoint.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/summary",
    ],
)

tpu_py_test(
    name = "async_checkpoint_test",
    size = "medium",
    srcs = ["async_checkpoint_test.py"],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":async_checkpoint",
        ":tpu_estimator",
        ":tpu_lib",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//third_party/py/numpy",
    ],
)

pytype_library(
    name = "device_assignment",
    srcs = ["device_assignment.py"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":topology",
        "//tensorflow/python:platform",
        "//tensorflow/python/util:tf_export",
    ],
)

pytype_library(
    name = "preempted_hook_py",
    srcs = ["preempted_hook.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:platform",
        "//tensorflow/python:session_run_hook",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
    ],
)

py_library(
    name = "tpu_replication",
    srcs = ["tpu_replication.py"],
    deps = [
        ":device_assignment",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/framework:c_api_util",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/types",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

py_library(
    name = "tpu_estimator",
    srcs = [
        "_tpu_estimator_embedding.py",
        "error_handling.py",
        "tpu_config.py",
        "tpu_context.py",
        "tpu_estimator.py",
        "util.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":async_checkpoint",
        ":feature_column",
        ":feature_column_v2",
        ":functional",
        ":preempted_hook_py",
        ":tpu_embedding",
        ":tpu_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:function",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/estimator:util",
        "//tensorflow/python/summary",
    ],
)

py_library(
    name = "functional",
    srcs = ["functional.py"],
    srcs_version = "PY3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/python:tpu_ops_gen",
    ],
)

pytype_library(
    name = "topology",
    srcs = ["topology.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core/protobuf/tpu:topology_proto_py",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tpu",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_column",
        ":feature_column_v2",
        ":tpu_embedding",
        ":tpu_estimator",
        ":tpu_lib",
    ],
)

py_library(
    name = "tpu_noestimator",
    srcs = [
        "__init__.py",
        "api.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_column",
        ":feature_column_v2",
        ":preempted_hook_py",
        ":tpu_embedding",
        ":tpu_embedding_base",
        ":tpu_embedding_for_serving",
        ":tpu_embedding_v1",
        ":tpu_embedding_v2",
        ":tpu_embedding_v2_utils",
        ":tpu_hardware_feature",
        ":tpu_lib",
    ],
)

pytype_library(
    name = "tensor_tracer",
    srcs = [
        "tensor_tracer.py",
        "tensor_tracer_flags.py",
        "tensor_tracer_report.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":tpu_replication",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_analytics",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/summary",
        "//tensorflow/python/tpu:tensor_tracer_proto_py",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/training:training_util",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
    ],
)

pytype_library(
    name = "tpu_lib",
    srcs = [
        "__init__.py",
        "bfloat16.py",
        "session_support.py",
        "tpu_optimizer.py",
        "tpu_strategy_util.py",
        "training_loop.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":datasets",
        ":device_assignment",
        ":functional",
        ":tensor_tracer",
        ":topology",
        ":tpu_feed",
        ":tpu_function",
        ":tpu_replication",
        ":tpu_sharding",
        "//tensorflow/compiler/xla/python_api:xla_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/protobuf/tpu:compilation_result_proto_py",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_py",
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_py",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:batch_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_analytics",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tf2",
        "//tensorflow/python:tpu_ops_gen",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/compiler/xla",
        "//tensorflow/python/compiler/xla/experimental:xla_sharding",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/tpu/profiler",
    ],
)

pytype_library(
    name = "tpu_py",
    srcs = ["tpu.py"],
    deps = [
        ":device_assignment",
        ":tensor_tracer",
        ":tpu_feed",
        ":tpu_function",
        ":tpu_name_util",
        ":tpu_replication",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_py",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:auto_control_deps",
        "//tensorflow/python:c_api_util",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:config",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/compiler/xla",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
    ],
)

pytype_library(
    name = "tpu_feed",
    srcs = ["tpu_feed.py"],
    deps = [
        ":tpu_name_util",
        ":tpu_sharding",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/compiler/xla/experimental:xla_sharding",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/user_ops:ops",
    ],
)

pytype_library(
    name = "tpu_function",
    srcs = ["tpu_function.py"],
)

pytype_library(
    name = "tpu_sharding",
    srcs = ["tpu_sharding.py"],
    deps = [
        "//tensorflow/python:tensor_shape",
    ],
)

pytype_library(
    name = "tpu_system_metadata",
    srcs = ["tpu_system_metadata.py"],
    deps = [
        ":tpu_py",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//tensorflow/python:config",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute:device_util",
    ],
)

pytype_library(
    name = "datasets",
    srcs = [
        "datasets.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:function",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/types",
    ],
)

tf_py_test(
    name = "datasets_test",
    size = "medium",
    srcs = ["datasets_test.py"],
    grpc_enabled = True,
    shard_count = 4,
    tags = ["no_oss"],
    deps = [
        ":datasets",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "tpu_test",
    size = "small",
    srcs = ["tpu_test.py"],
    tags = [
        "no_oss",  # TODO(b/131157871): Reenable in OSS when fixed
        "no_windows",  # TODO: needs investigation on Windows
    ],
    deps = [
        ":tpu",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:layers",
    ],
)

tf_py_test(
    name = "tpu_sharding_test",
    size = "small",
    srcs = ["tpu_sharding_test.py"],
    deps = [
        ":tpu_sharding",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
    ],
)

tf_py_test(
    name = "bfloat16_test",
    size = "small",
    srcs = ["bfloat16_test.py"],
    deps = [
        ":tpu",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
    ],
)

tf_py_test(
    name = "tpu_infeed_test",
    size = "small",
    srcs = ["tpu_infeed_test.py"],
    deps = [
        ":tpu",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "topology_test",
    size = "medium",
    srcs = ["topology_test.py"],
    deps = [
        ":topology",
        "//tensorflow/python:framework_test_lib",
    ],
)

pytype_library(
    name = "tpu_embedding",
    srcs = [
        "tpu_embedding.py",
        "tpu_embedding_gradient.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":tpu_lib",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:tpu_ops_gen",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

pytype_library(
    name = "tpu_strategy_util",
    srcs = ["tpu_strategy_util.py"],
    visibility = [
        "//learning/brain:__subpackages__",
        "//learning/deepmind:__subpackages__",
        "//learning/serving:__subpackages__",
        "//research/graph:__subpackages__",
        "//tensorflow:__subpackages__",
        "//third_party/py/tensorflow_numerics/extensions:__pkg__",
    ],
    deps = [
        ":tpu_lib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
    ],
)

pytype_library(
    name = "tpu_hardware_feature",
    srcs = ["tpu_hardware_feature.py"],
    deps = [
        "//tensorflow/core/protobuf/tpu:topology_proto_py",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "tpu_name_util",
    srcs = ["tpu_name_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

pytype_library(
    name = "feature_column",
    srcs = ["feature_column.py"],
    deps = [
        ":tpu_lib",
        ":tpu_py",
        ":tpu_replication",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/feature_column:feature_column_py",
    ],
)

pytype_library(
    name = "feature_column_v2",
    srcs = ["feature_column_v2.py"],
    deps = [
        ":feature_column",
        ":tpu_lib",
        ":tpu_py",
        ":tpu_replication",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/feature_column:feature_column_py",
    ],
)

tf_py_test(
    name = "feature_column_test",
    srcs = [
        "feature_column_test.py",
    ],
    main = "feature_column_test.py",
    deps = [
        ":feature_column",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/feature_column:feature_column_py",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "feature_column_v2_test",
    srcs = [
        "feature_column_v2_test.py",
    ],
    main = "feature_column_v2_test.py",
    tags = ["no_oss"],  # Due to the usage of keras component.
    deps = [
        ":feature_column_v2",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/feature_column:feature_column_py",
        "//third_party/py/numpy",
    ],
)

pytype_library(
    name = "tpu_embedding_v2_utils",
    srcs = ["tpu_embedding_v2_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_py",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/framework:device_spec",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/types",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

pytype_library(
    name = "tpu_embedding_v2",
    srcs = ["tpu_embedding_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_utils",
        ":tpu_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:distribute_utils",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_base",
    srcs = ["tpu_embedding_base.py"],
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_utils",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/util",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_for_serving",
    srcs = ["tpu_embedding_for_serving.py"],
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base",
        ":tpu_embedding_v2_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/types",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

tf_py_test(
    name = "tpu_embedding_for_serving_test",
    srcs = [
        "tpu_embedding_for_serving_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_for_serving",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v1",
    srcs = ["tpu_embedding_v1.py"],
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base",
        ":tpu_embedding_v2_utils",
        ":tpu_py",
        ":tpu_replication",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "tpu_embedding_v2_utils_test",
    srcs = [
        "tpu_embedding_v2_utils_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2",
        "//tensorflow/python/compat:v2_compat",
    ],
)

tpu_py_test(
    name = "tpu_outside_compilation_test",
    srcs = [
        "tpu_outside_compilation_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":tpu_lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
    ],
)

# NOTE this target should only be depended on by the tpu_test_wrapper macro.
py_library(
    name = "tpu_test_deps",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python:client_testlib"],
)

tf_proto_library(
    name = "tensor_tracer_proto",
    srcs = ["tensor_tracer.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "tensor_tracer_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":tensor_tracer_proto"],
# )
# copybara:uncomment_end
