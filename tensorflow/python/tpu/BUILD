load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Description: Operations defined for Cloud TPUs
load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

# Placeholder: load py_proto_library
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test", "tf_python_pybind_extension")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/core/platform:build_config_root.bzl", "if_pywrap")
load("//tensorflow/python/tpu:tpu.bzl", "internal_create_sanitizer_settings", "tpu_py_strict_test")

# Do not add anymore paths here. You do not need to be in the visibility list
# to use TPU symbols. They are accessible from tf.contrib.tpu in TF 1.x and
# tf.tpu and tf.compat.v1.tpu in TF 2.x.
visibility = [
    "//learning/brain:__subpackages__",
    "//learning/deepmind:__subpackages__",
    "//learning/serving:__subpackages__",
    "//research/graph:__subpackages__",
    "//smartass/brain/configure:__subpackages__",
    "//third_party/py/jax_tpu_embedding:__subpackages__",
    "//third_party/py/lingvo:__subpackages__",
    "//third_party/py/medical_research_foundations:__subpackages__",
    "//tensorflow:__subpackages__",
    "//waymo/ml/deploy/sync_test/tools:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility + ["//tensorflow/tools/pip_package:__subpackages__"],
    licenses = ["notice"],
)

exports_files(["tpu_test_wrapper.py"])

bzl_library(
    name = "tpu_test_wrapper_bzl",
    srcs = ["tpu_test_wrapper.bzl"],
)

py_strict_library(
    name = "tpu_test_wrapper",
    srcs = ["tpu_test_wrapper.py"],
    deps = [
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_strict_test(
    name = "tpu_test_wrapper_test",
    srcs = ["tpu_test_wrapper_test.py"],
    main = "tpu_test_wrapper_test.py",
    tags = [
        "no_oss_py35",
        "no_pip",
    ],
    deps = [
        ":tpu_test_wrapper",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:flags",
        "@absl_py//absl/testing:flagsaver",
    ],
)

tf_py_strict_test(
    name = "tpu_embedding_v3_cpu_ops_test",
    srcs = ["tpu_embedding_v3_cpu_ops_test.py"],
    shard_count = 8,
    tags = ["no_oss"],
    deps = [
        # copybara:uncomment "//tensorflow/core/tpu/kernels",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:bitwise_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tpu/ops:gen_xla_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

alias(
    name = "tpu_ops",
    actual = "//tensorflow/python/tpu/ops",
)

pytype_strict_library(
    name = "async_checkpoint",
    srcs = ["async_checkpoint.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:pywrap_saved_model",
        "//tensorflow/python/training",
        "//tensorflow/python/training:basic_session_run_hooks",
        "//tensorflow/python/training:monitored_session",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/training:session_run_hook",
        "//tensorflow/python/training:summary_io",
        "//tensorflow/python/training:training_util",
    ],
)

tpu_py_strict_test(
    name = "async_checkpoint_test",
    size = "medium",
    srcs = ["async_checkpoint_test.py"],
    disable_experimental = True,
    disable_mlir_bridge = False,
    tags = ["no_oss"],
    deps = [
        ":async_checkpoint",
        ":tpu_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:metrics",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:pywrap_saved_model",
        "//tensorflow/python/training:basic_session_run_hooks",
        "//tensorflow/python/training:training_lib",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "device_assignment",
    srcs = ["device_assignment.py"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":topology",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:numpy_compat",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "preempted_hook_py",
    srcs = ["preempted_hook.py"],
    deps = [
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:session_run_hook",
    ],
)

py_strict_library(
    name = "tpu_replication",
    srcs = ["tpu_replication.py"],
    deps = [
        ":device_assignment",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "functional",
    srcs = ["functional.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/python/tpu/ops",
    ],
)

pytype_strict_library(
    name = "topology",
    srcs = ["topology.py"],
    deps = [
        "//tensorflow/core/protobuf/tpu:topology_proto_py",
        "//tensorflow/python/util:numpy_compat",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "tpu",
    srcs = [
        "__init__.py",
    ],
    deps = ["//tensorflow/python/framework:versions"],
)

py_strict_library(
    name = "tpu_noestimator",
    srcs = [
        "__init__.py",
        "api.py",
    ],
    deps = [
        ":feature_column_v2",
        ":tpu_embedding_for_serving",
        ":tpu_embedding_v1",
        ":tpu_embedding_v2",
        ":tpu_embedding_v2_utils",
        ":tpu_embedding_v3",
        ":tpu_hardware_feature",
        ":tpu_lib",
        ":tpu_py",
        "//tensorflow/python/framework:versions",
    ],
)

pytype_strict_library(
    name = "tensor_tracer",
    srcs = ["tensor_tracer.py"],
    deps = [
        ":tensor_tracer_flags",
        ":tensor_tracer_report",
        ":tpu_replication",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:graph_io",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_case",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:logging_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_impl",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/platform:analytics",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:remote_utils",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/summary:summary_iterator",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/training:training_util",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "tensor_tracer_report",
    srcs = ["tensor_tracer_report.py"],
    deps = [
        ":tensor_tracer_proto_py",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
    ],
)

pytype_strict_library(
    name = "tensor_tracer_flags",
    srcs = ["tensor_tracer_flags.py"],
    deps = [
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:tf_logging",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_library(
    name = "tpu_lib",
    srcs = [
        "__init__.py",
        "bfloat16.py",
        "tpu_optimizer.py",
        "tpu_strategy_util.py",
        "training_loop.py",
    ],
    deps = [
        ":tensor_tracer",
        ":topology",
        ":tpu_feed",
        ":tpu_function",
        ":tpu_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/compiler/xla",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:versions",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/training:optimizer",
        "//tensorflow/python/training:session_run_hook",
        "//tensorflow/python/training:training_util",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "tpu_py",
    srcs = ["tpu.py"],
    deps = [
        ":device_assignment",
        ":tensor_tracer",
        ":tpu_feed",
        ":tpu_function",
        ":tpu_name_util",
        ":tpu_replication",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/protobuf/tpu:compilation_result_proto_py",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_py",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/compiler/xla",
        "//tensorflow/python/framework:auto_control_deps",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:traceback_utils",
        "//tensorflow/python/util:variable_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "tpu_feed",
    srcs = ["tpu_feed.py"],
    deps = [
        ":tpu_name_util",
        ":tpu_sharding",
        "//tensorflow/python/compiler/xla/experimental:xla_sharding",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "tpu_function",
    srcs = ["tpu_function.py"],
)

pytype_strict_library(
    name = "tpu_sharding",
    srcs = ["tpu_sharding.py"],
    deps = [
        "//tensorflow/python/framework:tensor_shape",
    ],
)

pytype_strict_library(
    name = "tpu_system_metadata",
    srcs = ["tpu_system_metadata.py"],
    deps = [
        ":tpu_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "datasets",
    srcs = [
        "datasets.py",
    ],
    visibility = visibility,
    deps = [
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:functional_ops",
        "//tensorflow/python/types:data",
    ],
)

tf_py_strict_test(
    name = "datasets_test",
    size = "medium",
    srcs = ["datasets_test.py"],
    grpc_enabled = True,
    shard_count = 4,
    tags = ["no_oss"],
    deps = [
        ":datasets",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/lib/io:python_io",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "tpu_test",
    size = "small",
    srcs = ["tpu_test.py"],
    tags = [
        "no_oss",  # TODO(b/131157871): Reenable in OSS when fixed
        "no_windows",  # TODO: needs investigation on Windows
    ],
    deps = [
        ":tpu_feed",
        ":tpu_lib",
        ":tpu_py",
        ":tpu_replication",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/layers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tpu/ops",
    ],
)

tf_py_strict_test(
    name = "tpu_sharding_test",
    size = "small",
    srcs = ["tpu_sharding_test.py"],
    deps = [
        ":tpu_sharding",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "bfloat16_test",
    size = "small",
    srcs = ["bfloat16_test.py"],
    deps = [
        ":tpu_lib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "tpu_infeed_test",
    size = "small",
    srcs = ["tpu_infeed_test.py"],
    deps = [
        ":tpu_feed",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "topology_test",
    size = "medium",
    srcs = ["topology_test.py"],
    deps = [
        ":topology",
        "//tensorflow/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "tpu_strategy_util",
    srcs = ["tpu_strategy_util.py"],
    visibility = [
        "//tensorflow:__subpackages__",
        "//third_party/py/tensorflow_numerics/extensions:__pkg__",
    ],
    deps = [
        ":topology",
        ":tpu_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
    ],
)

pytype_strict_library(
    name = "tpu_hardware_feature",
    srcs = ["tpu_hardware_feature.py"],
    deps = [
        "//tensorflow/core/protobuf/tpu:topology_proto_py",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "tpu_name_util",
    srcs = ["tpu_name_util.py"],
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "feature_column",
    srcs = ["feature_column.py"],
    deps = [
        ":tpu_function",
        ":tpu_py",
        ":tpu_replication",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:variable_scope",
    ],
)

pytype_strict_library(
    name = "feature_column_v2",
    srcs = ["feature_column_v2.py"],
    deps = [
        ":feature_column",
        ":tpu_py",
        ":tpu_replication",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "feature_column_test",
    srcs = [
        "feature_column_test.py",
    ],
    main = "feature_column_test.py",
    deps = [
        ":feature_column",
        "//tensorflow/python/client:session",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "feature_column_v2_test",
    srcs = [
        "feature_column_v2_test.py",
    ],
    main = "feature_column_v2_test.py",
    tags = ["no_oss"],  # Due to the usage of keras component.
    deps = [
        ":feature_column_v2",
        ":tpu_function",
        ":tpu_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v2_utils",
    srcs = ["tpu_embedding_v2_utils.py"],
    deps = [
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_py",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/framework:device_spec",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops_v2",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v2",
    srcs = ["tpu_embedding_v2.py"],
    deps = [
        ":tpu_embedding_v2_utils",
        ":tpu_py",
        ":tpu_replication",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
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
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_base",
    srcs = ["tpu_embedding_base.py"],
    deps = [
        ":tpu_embedding_v2_utils",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/util:nest",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v3_utils",
    srcs = ["tpu_embedding_v3_utils.py"],
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops_gen",
        "//tensorflow/python/ops:manip_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/trackable:base",
    ],
)

tf_py_strict_test(
    name = "tpu_embedding_v3_utils_test",
    srcs = ["tpu_embedding_v3_utils_test.py"],
    deps = [
        ":tpu_embedding_v3_utils",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v3_checkpoint_adapter",
    srcs = ["tpu_embedding_v3_checkpoint_adapter.py"],
    deps = [
        ":tpu_embedding_v3_utils",
        "//tensorflow/core/tpu/kernels:sparse_core_layout_proto_py",
        "//tensorflow/python/checkpoint:checkpoint_adapter",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training:py_checkpoint_reader",
        "//tensorflow/python/util/protobuf",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "tpu_embedding_v3_checkpoint_adapter_test",
    srcs = ["tpu_embedding_v3_checkpoint_adapter_test.py"],
    deps = [
        ":tpu_embedding_v3_checkpoint_adapter",
        "//tensorflow/core/tpu/kernels:sparse_core_layout_proto_py",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_for_serving",
    srcs = ["tpu_embedding_for_serving.py"],
    deps = [
        ":tpu_embedding_base",
        ":tpu_embedding_v2_utils",
        ":tpu_embedding_v3_utils",
        "//tensorflow/core/tpu/kernels:sparse_core_layout_proto_py",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:constants",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "tpu_embedding_for_serving_test",
    srcs = [
        "tpu_embedding_for_serving_test.py",
    ],
    deps = [
        ":tpu_embedding_for_serving",
        ":tpu_embedding_v2_utils",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops:init_ops_v2",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v1",
    srcs = ["tpu_embedding_v1.py"],
    deps = [
        ":tpu_embedding_base",
        ":tpu_embedding_v2_utils",
        ":tpu_replication",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "tpu_embedding_v2_utils_test",
    srcs = [
        "tpu_embedding_v2_utils_test.py",
    ],
    deps = [
        ":tpu_embedding_v2_utils",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops_v2",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_outside_compilation_test",
    srcs = [
        "tpu_outside_compilation_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":functional",
        ":tpu_py",
        ":tpu_replication",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:image_ops",
        "//tensorflow/python/ops:logging_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/tpu/ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

# NOTE this target should only be depended on by the tpu_test_wrapper macro.
py_strict_library(
    name = "tpu_test_deps",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/util:tf_inspect",
    ],
)

tf_proto_library(
    name = "tensor_tracer_proto",
    srcs = ["tensor_tracer.proto"],
    protodeps = [
        "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "tensor_tracer_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":tensor_tracer_proto"],
# )
# copybara:uncomment_end

tf_python_pybind_extension(
    name = "_pywrap_sparse_core_layout",
    srcs = ["pywrap_sparse_core_layout.cc"],
    enable_stub_generation = True,
    features = ["-layering_check"],
    pytype_srcs = [
        "_pywrap_sparse_core_layout.pyi",
    ],
    deps = [
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:pybind11_status_headers",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_pywrap(
        if_false = [
            "//tensorflow/core/tpu/kernels:_pywrap_sparse_core_layout_header_only",
        ],
        if_true = [
            "//tensorflow/core/tpu/kernels:sparse_core_layout",
        ],
    ),
)

tf_python_pybind_extension(
    name = "_pywrap_tpu_embedding",
    srcs = ["pywrap_tpu_embedding.cc"],
    enable_stub_generation = True,
    features = ["-layering_check"],
    pytype_srcs = [
        "_pywrap_tpu_embedding.pyi",
    ],
    deps = [
        "//tensorflow/core/tpu/kernels:sparse_core_ops_utils",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

py_strict_library(
    name = "tpu_embedding_v3",
    srcs = ["tpu_embedding_v3.py"],
    deps = [
        ":_pywrap_sparse_core_layout",
        ":_pywrap_tpu_embedding",
        ":tpu_embedding_base",
        ":tpu_embedding_v2_utils",
        ":tpu_embedding_v3_checkpoint_adapter",
        ":tpu_embedding_v3_utils",
        ":tpu_replication",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/tpu/kernels:sparse_core_layout_proto_py",
        "//tensorflow/python/checkpoint:saveable_compat",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/distribute:tpu_util",
        "//tensorflow/python/distribute:tpu_values",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/distribute:values_util",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops_gen",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/tpu/ops:gen_xla_ops",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "@absl_py//absl/logging",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v3_test",
    srcs = ["tpu_embedding_v3_test.py"],
    disable_experimental = True,
    disable_tfrt = False,
    disable_v2 = True,
    disable_v3 = True,
    # copybara:uncomment disable_v4i = True,
    # copybara:uncomment disable_v5 = False,
    # copybara:uncomment disable_v5_grm = True,
    # copybara:uncomment disable_v6e = True,
    shard_count = 4,
    # copybara:uncomment use_tpu_plugin_and_capi = True,
    deps = [
        ":tpu_embedding_for_serving",
        ":tpu_embedding_v2_utils",
        ":tpu_embedding_v3",
        ":tpu_replication",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops_v2",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v3_additional_test",
    srcs = ["tpu_embedding_v3_additional_test.py"],
    args = [
        "--tpu=''",
    ],
    disable_experimental = True,
    disable_tfrt = False,
    disable_v2 = True,
    disable_v3 = True,
    # copybara:uncomment disable_v4i = True,
    # copybara:uncomment disable_v5 = False,
    # copybara:uncomment disable_v5_grm = True,
    env = {
        "TF_XLA_FLAGS": (
            "--tf_mlir_enable_mlir_bridge=true " +
            "--tf_mlir_enable_convert_control_to_data_outputs_pass=true " +
            "--tf_mlir_enable_merge_control_flow_pass=true " +
            "--tf_mlir_enable_strict_clusters=true " +
            # These tests run lookup without while loops.
            "--tf_xla_disable_full_embedding_pipelining=true"
        ),
    },
    # Make it so there are enough shards so each test runs in separate shards.
    shard_count = 12,
    visibility = ["//visibility:private"],
    deps = [
        ":device_assignment",
        ":tpu_embedding_for_serving",
        ":tpu_embedding_v2_utils",
        ":tpu_embedding_v3",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

internal_create_sanitizer_settings()
