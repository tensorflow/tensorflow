# Description: Operations defined for Cloud TPUs

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//cloud/vmm/testing/tests/tpu:__subpackages__",
        "//knowledge/cerebra/sense/im2query:__subpackages__",
        "//learning/brain:__subpackages__",
        "//learning/deepmind:__subpackages__",
        "//medical/pathology:__subpackages__",
        "//tensorflow:__subpackages__",
        "//vr/perception:__subpackages__",
    ],
)

cc_library(
    name = "all_ops",
    deps = [
        ":cross_replica_ops_op_lib",
        ":heartbeat_ops_op_lib",
        ":host_compute_ops_op_lib",
        ":infeed_ops_op_lib",
        ":outfeed_ops_op_lib",
        ":replication_ops_op_lib",
        ":tpu_configuration_ops_op_lib",
        ":tpu_embedding_ops_op_lib",
    ],
)

py_library(
    name = "async_checkpoint",
    srcs = ["python/tpu/async_checkpoint.py"],
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
        "python/tpu/error_handling.py",
        "python/tpu/tpu_config.py",
        "python/tpu/tpu_context.py",
        "python/tpu/tpu_estimator.py",
        "python/tpu/util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":async_checkpoint",
        ":tpu_lib",
        "//tensorflow/contrib/training:training_py",
        "//tensorflow/core:protos_all_py",
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
        "//tensorflow/python/estimator:util",
        "@six_archive//:six",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "cross_replica_ops",
        "heartbeat_ops",
        "host_compute_ops",
        "infeed_ops",
        "outfeed_ops",
        "replication_ops",
        "tpu_configuration_ops",
        "tpu_embedding_ops",
    ],
    deps = [
        "//tensorflow/contrib/tpu/proto:tpu_embedding_configuration_proto_cc",
        "//tensorflow/contrib/tpu/utils:tpu_embedding_optimization_parameters_utils",
        "//tensorflow/contrib/tpu/utils:tpu_embedding_output_layout_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_custom_op_library(
    name = "python/ops/_tpu_ops.so",
    srcs = [
        "ops/cross_replica_ops.cc",
        "ops/heartbeat_ops.cc",
        "ops/host_compute_ops.cc",
        "ops/infeed_ops.cc",
        "ops/outfeed_ops.cc",
        "ops/replication_ops.cc",
        "ops/tpu_configuration_ops.cc",
        "ops/tpu_embedding_ops.cc",
    ],
    deps = [
        "//tensorflow/contrib/tpu/proto:tpu_embedding_configuration_proto_cc",
        "//tensorflow/contrib/tpu/utils:tpu_embedding_optimization_parameters_utils",
        "//tensorflow/contrib/tpu/utils:tpu_embedding_output_layout_utils",
        "//tensorflow/core:lib_proto_parsing",
    ],
)

tf_gen_op_wrapper_py(
    name = "tpu_ops",
    hidden = [
        "SendTPUEmbeddingGradients",
        "EnqueueTPUEmbeddingIntegerBatch",
        "EnqueueTPUEmbeddingSparseBatch",
        "EnqueueTPUEmbeddingSparseTensorBatch",
    ],
    deps = [
        ":cross_replica_ops_op_lib",
        ":heartbeat_ops_op_lib",
        ":host_compute_ops_op_lib",
        ":infeed_ops_op_lib",
        ":outfeed_ops_op_lib",
        ":replication_ops_op_lib",
        ":tpu_configuration_ops_op_lib",
        ":tpu_embedding_ops_op_lib",
    ],
)

py_library(
    name = "profiler",
    srcs = ["python/profiler/__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/tpu/profiler:tpu_profiler_analysis_pb2_grpc",
        "//tensorflow/contrib/tpu/profiler:tpu_profiler_analysis_proto_py",
        "//tensorflow/contrib/tpu/profiler:trace_events_proto_py",
        "//tensorflow/python:util",
    ],
)

tf_custom_op_py_library(
    name = "tpu_py",
    srcs = glob(["python/ops/*.py"]),
    dso = [":python/ops/_tpu_ops.so"],
    kernels = [
        ":all_ops",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":profiler",
        ":tpu_ops",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "tpu",
    srcs = [
        "__init__.py",
        "python/tpu/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":keras_support",  # split out to avoid cycle with tpu_strategy
        ":tpu_embedding",
        ":tpu_estimator",
        ":tpu_lib",
    ],
)

py_library(
    name = "keras_support",
    srcs = [
        "python/tpu/keras_support.py",
        "python/tpu/keras_tpu_variables.py",
    ],
    srcs_version = "PY2AND3",
    visibility = [
        "//cloud/vmm/testing/tests/tpu:__subpackages__",
        "//learning/brain:__subpackages__",
        "//tensorflow:__subpackages__",
        "//third_party/cloud_tpu/models/keras_colab:__subpackages__",
        "//third_party/cloud_tpu/models/mnist_keras:__subpackages__",
        "//third_party/cloud_tpu/models/resnet50_keras:__subpackages__",
    ],
    deps = [
        ":tpu_lib",
        "//tensorflow/contrib/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/contrib/distribute",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/tpu/proto:compilation_result_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:layers",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tpu_lib",
    srcs = [
        "python/tpu/__init__.py",
        "python/tpu/bfloat16.py",
        "python/tpu/device_assignment.py",
        "python/tpu/session_support.py",
        "python/tpu/tensor_tracer.py",
        "python/tpu/topology.py",
        "python/tpu/tpu.py",
        "python/tpu/tpu_feed.py",
        "python/tpu/tpu_function.py",
        "python/tpu/tpu_optimizer.py",
        "python/tpu/tpu_sharding.py",
        "python/tpu/tpu_system_metadata.py",
        "python/tpu/training_loop.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":datasets",
        ":profiler",
        ":tpu_py",
        "//tensorflow/compiler/xla/experimental/xla_sharding",
        "//tensorflow/compiler/xla/python_api:xla_shape",
        "//tensorflow/contrib/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/contrib/tpu/proto:compilation_result_proto_py",
        "//tensorflow/contrib/tpu/proto:optimization_parameters_proto_py",
        "//tensorflow/contrib/tpu/proto:topology_proto_py",
        "//tensorflow/contrib/tpu/proto:tpu_embedding_configuration_proto_py",
        "//tensorflow/contrib/tpu/proto:tpu_embedding_output_layout_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/ops/losses",
    ],
)

py_library(
    name = "datasets",
    srcs = [
        "python/tpu/datasets.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/ops:batching",
        "//tensorflow/contrib/data/python/ops:interleave_ops",
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
    srcs = ["python/tpu/datasets_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        ":datasets",
    ],
    flaky = 1,  # TODO(b/117363808): fails 1/1000 OSS runs
    grpc_enabled = True,
)

tf_py_test(
    name = "tpu_test",
    size = "small",
    srcs = ["python/tpu/tpu_test.py"],
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
    srcs = ["python/tpu/tpu_sharding_test.py"],
    additional_deps = [
        ":tpu",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
    ],
)

tf_py_test(
    name = "bfloat16_test",
    size = "small",
    srcs = ["python/tpu/bfloat16_test.py"],
    additional_deps = [
        ":tpu",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
    ],
)

tf_py_test(
    name = "tpu_infeed_test",
    size = "small",
    srcs = ["python/tpu/tpu_infeed_test.py"],
    additional_deps = [
        ":tpu",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "tpu_config_test",
    size = "small",
    srcs = ["python/tpu/tpu_config_test.py"],
    additional_deps = [
        ":tpu_estimator",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "tpu_estimator_signals_test",
    size = "small",
    srcs = ["python/tpu/tpu_estimator_signals_test.py"],
    additional_deps = [
        ":tpu_estimator",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "topology_test",
    size = "medium",
    srcs = ["python/tpu/topology_test.py"],
    additional_deps = [
        ":tpu",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_library(
    name = "tpu_embedding",
    srcs = ["python/tpu/tpu_embedding.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/tpu:tpu_ops",
        "//tensorflow/contrib/tpu/proto:tpu_embedding_configuration_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "@six_archive//:six",
    ],
)
