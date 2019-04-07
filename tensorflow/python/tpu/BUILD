# Description: Operations defined for Cloud TPUs

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_py_test",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//cloud/vmm/testing/tests/tpu:__subpackages__",
        "//knowledge/cerebra/sense/im2query:__subpackages__",
        "//learning/brain:__subpackages__",
        "//learning/deepmind:__subpackages__",
        "//medical/pathology:__subpackages__",
        "//research/graph:__subpackages__",
        "//tensorflow:__subpackages__",
        "//vr/perception:__subpackages__",
    ],
)

py_library(
    name = "tpu_py",
    srcs = ["ops/tpu_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:tpu_ops_gen",
    ],
)

py_library(
    name = "async_checkpoint",
    srcs = ["async_checkpoint.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
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
    srcs_version = "PY2AND3",
    deps = [
        ":async_checkpoint",
        ":feature_column",
        ":functional",
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
        "//tensorflow/python:summary",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/estimator:util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "functional",
    srcs = ["functional.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/python:tpu_ops_gen",
    ],
)

py_library(
    name = "tpu",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_column",
        ":tpu_embedding",
        ":tpu_estimator",
        ":tpu_lib",
    ],
)

py_library(
    name = "tpu_lib",
    srcs = [
        "__init__.py",
        "bfloat16.py",
        "device_assignment.py",
        "profile_logger.py",
        "session_support.py",
        "tensor_tracer.py",
        "topology.py",
        "tpu.py",
        "tpu_feed.py",
        "tpu_function.py",
        "tpu_optimizer.py",
        "tpu_sharding.py",
        "tpu_strategy_util.py",
        "tpu_system_metadata.py",
        "training_loop.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":datasets",
        ":functional",
        ":tpu_py",
        "//tensorflow/compiler/xla/experimental/xla_sharding",
        "//tensorflow/compiler/xla/python_api:xla_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/protobuf/tpu:compilation_result_proto_py",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_py",
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_py",
        "//tensorflow/core/protobuf/tpu:topology_proto_py",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_output_layout_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tpu_ops_gen",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/compiler/xla",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/tpu/profiler",
    ],
)

py_library(
    name = "datasets",
    srcs = [
        "datasets.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:function",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:readers",
    ],
)

tf_py_test(
    name = "datasets_test",
    size = "medium",
    srcs = ["datasets_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        ":datasets",
    ],
    grpc_enabled = True,
    shard_count = 4,
    tags = ["no_oss"],
)

tf_py_test(
    name = "tpu_test",
    size = "small",
    srcs = ["tpu_test.py"],
    additional_deps = [
        ":tpu",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:layers",
    ],
    tags = ["no_windows"],  # TODO: needs investigation on Windows
)

tf_py_test(
    name = "tpu_sharding_test",
    size = "small",
    srcs = ["tpu_sharding_test.py"],
    additional_deps = [
        ":tpu",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
    ],
)

tf_py_test(
    name = "bfloat16_test",
    size = "small",
    srcs = ["bfloat16_test.py"],
    additional_deps = [
        ":tpu",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
    ],
)

tf_py_test(
    name = "tpu_infeed_test",
    size = "small",
    srcs = ["tpu_infeed_test.py"],
    additional_deps = [
        ":tpu",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "tpu_config_test",
    size = "small",
    srcs = ["tpu_config_test.py"],
    additional_deps = [
        ":tpu_estimator",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "tpu_estimator_signals_test",
    size = "small",
    srcs = ["tpu_estimator_signals_test.py"],
    additional_deps = [
        ":tpu_estimator",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
    ],
    # TODO(jhseu): Remove. Fails in OSS on Python 3.
    tags = ["no_oss"],
)

tf_py_test(
    name = "topology_test",
    size = "medium",
    srcs = ["topology_test.py"],
    additional_deps = [
        ":tpu",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_library(
    name = "tpu_embedding",
    srcs = [
        "tpu_embedding.py",
        "tpu_embedding_gradient.py",
    ],
    srcs_version = "PY2AND3",
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
        "@six_archive//:six",
    ],
)

py_library(
    name = "tpu_strategy_util",
    srcs = ["tpu_strategy_util.py"],
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

py_library(
    name = "feature_column",
    srcs = ["feature_column.py"],
    deps = [
        ":tpu_lib",
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
    additional_deps = [
        ":feature_column",
        "//third_party/py/numpy",
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
    ],
    main = "feature_column_test.py",
)
