load("//tensorflow:tensorflow.bzl", "py_test", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow/core/platform/default:distribute.bzl", "distribute_py_test")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "distribute_test_lib_pip",
    deps = [
        ":combinations",
        ":model_combinations",
        ":multi_worker_test_base",
        ":saved_model_test_base",
        ":single_loss_example",
        ":strategy_combinations",
        ":strategy_test_lib",
        "//tensorflow/python/keras/distribute:keras_correctness_test_lib",
        "//tensorflow/python/keras/distribute:keras_test_lib",
    ],
)

py_library(
    name = "all_reduce",
    srcs = [
        "all_reduce.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nccl_ops",
    ],
)

tf_py_test(
    name = "all_reduce_test",
    srcs = ["all_reduce_test.py"],
    deps = [
        ":all_reduce",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "cross_device_ops",
    srcs = ["cross_device_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":collective_util",
        ":cross_device_utils",
        ":device_util",
        ":ps_values",
        ":reduce_util",
        ":tpu_values",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:device_lib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:kernels",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:tf_export",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:executor",
        "//tensorflow/tools/docs:doc_controls",
        "@six_archive//:six",
    ],
)

py_library(
    name = "cross_device_utils",
    srcs = ["cross_device_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":all_reduce",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nccl_ops",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "device_util",
    srcs = ["device_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:device",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "device_util_test",
    srcs = ["device_util_test.py"],
    deps = [
        ":device_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
    ],
)

py_library(
    name = "distribute",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cross_device_ops",
        ":distribute_lib",
        ":mirrored_strategy",
        ":one_device_strategy",
        ":sharded_variable",
        "//tensorflow/python/distribute/experimental",
    ],
)

py_library(
    name = "distribute_lib",
    srcs = [
        "distribute_lib.py",
        "distribution_strategy_context.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":collective_util",
        ":device_util",
        ":numpy_dataset",
        ":reduce_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/ops/losses:loss_reduction",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_test(
    name = "distribute_lib_test",
    size = "small",
    srcs = ["distribute_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = ["no_rocm"],
    deps = [
        ":combinations",
        ":distribute_lib",
        ":input_lib",
        ":reduce_util",
        ":values",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "distribute_config",
    srcs = [
        "distribute_config.py",
    ],
    deps = [],
)

py_library(
    name = "distribute_coordinator",
    srcs = [
        "distribute_coordinator.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":distribute_coordinator_context",
        ":multi_worker_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training_server_lib",
    ],
)

py_test(
    name = "distribute_coordinator_test",
    size = "medium",
    srcs = ["distribute_coordinator_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_oss_py2",  # b/138443278
    ],
    deps = [
        ":distribute_coordinator",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "distribute_coordinator_context",
    srcs = [
        "distribute_coordinator_context.py",
    ],
    srcs_version = "PY2AND3",
    deps = [],
)

py_library(
    name = "mirrored_run",
    srcs = ["mirrored_run.py"],
    deps = [
        ":device_util",
        ":distribute_lib",
        ":reduce_util",
        ":shared_variable_creator",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:config",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:device",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:tf_export",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/impl",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
    ],
)

py_library(
    name = "mirrored_strategy",
    srcs = ["mirrored_strategy.py"],
    deps = [
        ":cross_device_ops",
        ":device_util",
        ":distribute_lib",
        ":input_lib",
        ":mirrored_run",
        ":multi_worker_util",
        ":numpy_dataset",
        ":reduce_util",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
    ],
)

py_library(
    name = "parameter_server_strategy",
    srcs = ["parameter_server_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":cross_device_ops",
        ":device_util",
        ":distribute_lib",
        ":input_lib",
        ":mirrored_run",
        ":multi_worker_util",
        ":numpy_dataset",
        ":ps_values",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tf_export",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "central_storage_strategy",
    srcs = ["central_storage_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":device_util",
        ":distribute_lib",
        ":parameter_server_strategy",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "one_device_strategy",
    srcs = ["one_device_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":distribute_lib",
        ":input_lib",
        ":numpy_dataset",
        ":reduce_util",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

py_library(
    name = "collective_all_reduce_strategy",
    srcs = ["collective_all_reduce_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":cross_device_ops",
        ":cross_device_utils",
        ":input_lib",
        ":mirrored_strategy",
        ":multi_worker_util",
        ":numpy_dataset",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "mirrored_function_strategy",
    srcs = ["mirrored_function_strategy.py"],
    deps = [
        ":device_util",
        ":distribute_lib",
        ":mirrored_strategy",
        ":numpy_dataset",
        ":values",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
    ],
)

tf_py_test(
    name = "mirrored_function_strategy_test",
    srcs = ["mirrored_function_strategy_test.py"],
    tags = ["no_pip"],
    deps = [
        ":distribute_lib",
        ":mirrored_function_strategy",
        ":strategy_combinations",
        ":values",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "multi_worker_util",
    srcs = [
        "multi_worker_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:training_server_lib",
    ],
)

cuda_py_test(
    name = "multi_worker_continuous_run_test",
    srcs = [
        "multi_worker_continuous_run_test.py",
    ],
    tags = [
        "notsan",  # TODO(b/151841995)
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":multi_process_runner",
        ":multi_worker_test_base",
        ":reduce_util",
        ":strategy_combinations",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "numpy_dataset",
    srcs = ["numpy_dataset.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "numpy_dataset_test",
    size = "small",
    srcs = ["numpy_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":numpy_dataset",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "input_lib",
    srcs = ["input_lib.py"],
    deps = [
        ":device_util",
        ":distribute_lib",
        ":input_ops",
        ":values",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:multi_device_iterator_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_library(
    name = "input_ops",
    srcs = ["input_ops.py"],
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/util:nest",
    ],
)

cuda_py_test(
    name = "input_ops_test",
    srcs = ["input_ops_test.py"],
    deps = [
        ":input_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:structure",
    ],
)

py_test(
    name = "multi_worker_util_test",
    srcs = ["multi_worker_util_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":multi_worker_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "tpu_strategy",
    srcs = ["tpu_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":cross_device_ops",
        ":device_util",
        ":distribute_lib",
        ":input_lib",
        ":numpy_dataset",
        ":reduce_util",
        ":tpu_values",
        ":values",
        "//tensorflow/compiler/xla/experimental/xla_sharding",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:device_spec",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/impl",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/tpu:tpu_lib",
        "//tensorflow/python/tpu:tpu_py",
        "//third_party/py/numpy",
    ],
)

tpu_py_test(
    name = "tpu_strategy_test",
    srcs = ["tpu_strategy_test.py"],
    disable_experimental = True,
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":tpu_strategy",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras",
    ],
)

# Used only by estimator.
py_library(
    name = "estimator_training",
    srcs = [
        "estimator_training.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":distribute_coordinator",
        ":distribute_coordinator_context",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "reduce_util",
    srcs = ["reduce_util.py"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "collective_util",
    srcs = ["collective_util.py"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "shared_variable_creator",
    srcs = ["shared_variable_creator.py"],
)

py_test(
    name = "shared_variable_creator_test",
    srcs = ["shared_variable_creator_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":shared_variable_creator",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "summary_op_util",
    srcs = ["summary_op_util.py"],
    deps = [
        ":distribute_lib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_util",
    ],
)

py_library(
    name = "values",
    srcs = ["values.py"],
    deps = [
        ":device_util",
        ":distribute_lib",
        ":reduce_util",
        ":values_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:tf_export",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/training/tracking:base",
        "//tensorflow/python/types",
    ],
)

py_library(
    name = "ps_values",
    srcs = ["ps_values.py"],
    deps = [
        ":distribute_lib",
        ":values",
        ":values_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/training/tracking:base",
        "//tensorflow/python/types",
    ],
)

py_library(
    name = "values_util",
    srcs = ["values_util.py"],
    deps = [
        ":distribute_lib",
        ":reduce_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "tpu_values",
    srcs = ["tpu_values.py"],
    deps = [
        ":values",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/tpu:tpu_lib",
    ],
)

py_library(
    name = "combinations",
    srcs = ["combinations.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    deps = [
        ":multi_process_runner",
        ":multi_worker_test_base",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python:framework_test_combinations_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

py_test(
    name = "combinations_test",
    srcs = ["combinations_test.py"],
    tags = [
        "notap",  # TODO(b/153646955): flaky
    ],
    deps = [
        ":combinations",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python/distribute/cluster_resolver:tfconfig_cluster_resolver_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "strategy_combinations",
    srcs = ["strategy_combinations.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    deps = [
        ":central_storage_strategy",
        ":collective_all_reduce_strategy",
        ":combinations",
        ":distribute_lib",
        ":mirrored_strategy",
        ":multi_process_runner",
        ":multi_worker_test_base",
        ":one_device_strategy",
        ":tpu_strategy",
        "//tensorflow/python:config",
        "//tensorflow/python:platform",
        "//tensorflow/python:tf2",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/tpu:tpu_lib",
    ],
)

py_test(
    name = "strategy_combinations_test",
    srcs = ["strategy_combinations_test.py"],
    python_version = "PY3",
    deps = [
        ":combinations",
        ":reduce_util",
        ":strategy_combinations",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:config",
        "//tensorflow/python:constant_op",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "model_combinations",
    srcs = ["model_combinations.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":combinations",
        ":simple_models",
    ],
)

py_library(
    name = "multi_worker_test_base",
    srcs = ["multi_worker_test_base.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:remote",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "checkpoint_utils_test",
    size = "medium",
    srcs = ["checkpoint_utils_test.py"],
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

distribute_py_test(
    name = "checkpointing_test",
    srcs = ["checkpointing_test.py"],
    main = "checkpointing_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        ":tpu_strategy",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras",
        "//tensorflow/python/training/tracking:util",
    ],
)

distribute_py_test(
    name = "input_lib_test",
    srcs = ["input_lib_test.py"],
    main = "input_lib_test.py",
    shard_count = 10,
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":input_lib",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":reduce_util",
        ":strategy_combinations",
        ":values",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "input_lib_type_spec_test",
    srcs = ["input_lib_type_spec_test.py"],
    main = "input_lib_type_spec_test.py",
    shard_count = 10,
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":input_lib",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":reduce_util",
        ":strategy_combinations",
        ":values",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "cross_device_utils_test",
    srcs = ["cross_device_utils_test.py"],
    deps = [
        ":combinations",
        ":cross_device_utils",
        ":strategy_combinations",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "cross_device_ops_test",
    srcs = ["cross_device_ops_test.py"],
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":collective_util",
        ":combinations",
        ":cross_device_ops",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":strategy_combinations",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "one_device_strategy_test",
    srcs = ["one_device_strategy_test.py"],
    grpc_enabled = True,
    deps = [
        ":combinations",
        ":strategy_combinations",
        ":strategy_test_lib",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "sharded_variable",
    srcs = ["sharded_variable.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variables",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/training/tracking:base",
    ],
)

tf_py_test(
    name = "sharded_variable_test",
    size = "small",
    srcs = ["sharded_variable_test.py"],
    deps = [
        ":sharded_variable",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/training/tracking:util",
    ],
)

py_library(
    name = "strategy_test_lib",
    srcs = ["strategy_test_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":distribute_lib",
        ":reduce_util",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras/layers",
        "//third_party/py/numpy",
    ],
)

distribute_py_test(
    name = "values_test",
    size = "medium",
    srcs = ["values_test.py"],
    main = "values_test.py",
    shard_count = 5,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",
    ],
    tpu_tags = [
        "no_oss",  # b/150954621 Target too big to run serially reliably.
    ],
    deps = [
        ":combinations",
        ":distribute_lib",
        ":strategy_combinations",
        ":tpu_strategy",
        ":tpu_values",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:indexed_slices",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tf2",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model/model_utils:mode_keys",
        "//tensorflow/python/tpu:tpu_lib",
        "//tensorflow/python/types",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "ps_values_test",
    size = "medium",
    srcs = ["ps_values_test.py"],
    main = "ps_values_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":ps_values",
        ":strategy_combinations",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "moving_averages_test",
    srcs = ["moving_averages_test.py"],
    main = "moving_averages_test.py",
    deps = [
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "custom_training_loop_gradient_test",
    srcs = ["custom_training_loop_gradient_test.py"],
    main = "custom_training_loop_gradient_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/python:errors",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "custom_training_loop_input_test",
    srcs = ["custom_training_loop_input_test.py"],
    main = "custom_training_loop_input_test.py",
    shard_count = 5,
    tags = [
        "multi_and_single_gpu",
    ],
    tpu_tags = [
        "no_oss",  # Target too big to run serially reliably.
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/python:errors",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "custom_training_loop_metrics_test",
    srcs = ["custom_training_loop_metrics_test.py"],
    main = "custom_training_loop_metrics_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/python:errors",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "custom_training_loop_models_test",
    srcs = ["custom_training_loop_models_test.py"],
    main = "custom_training_loop_models_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    tpu_tags = [
        "no_oss",  # b/153615544.
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/python:errors",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "custom_training_loop_optimizer_test",
    srcs = ["custom_training_loop_optimizer_test.py"],
    main = "custom_training_loop_optimizer_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":distribute_lib",
        ":reduce_util",
        ":strategy_combinations",
        ":values",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "strategy_reduce_test",
    srcs = ["strategy_reduce_test.py"],
    main = "strategy_reduce_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/python:errors",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "minimize_loss_test",
    srcs = ["minimize_loss_test.py"],
    main = "minimize_loss_test.py",
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/139815303): enable after this is fixed.
        "noguitar",  # TODO(b/140755528): enable after this is fixed.
        "notap",  # TODO(b/139815303): enable after this is fixed.
    ],
    deps = [
        ":combinations",
        ":mirrored_strategy",
        ":single_loss_example",
        ":strategy_combinations",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/ops/losses",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "single_loss_example",
    srcs = ["single_loss_example.py"],
    deps = [
        ":step_fn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/optimizer_v2",
    ],
)

py_library(
    name = "step_fn",
    srcs = ["step_fn.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:training",
        "//tensorflow/python/eager:backprop",
    ],
)

distribute_py_test(
    name = "step_fn_test",
    srcs = ["step_fn_test.py"],
    main = "step_fn_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":single_loss_example",
        ":strategy_combinations",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "warm_starting_util_test",
    size = "medium",
    srcs = ["warm_starting_util_test.py"],
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "remote_mirrored_strategy_eager_test",
    srcs = ["remote_mirrored_strategy_eager_test.py"],
    tags = ["no_oss"],  # b/154743849
    deps = [
        ":combinations",
        ":distribute_lib",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":strategy_test_lib",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
)

cuda_py_test(
    name = "mirrored_strategy_test",
    srcs = ["mirrored_strategy_test.py"],
    shard_count = 5,
    tags = [
        "multi_and_single_gpu",
        "no_windows_gpu",  # TODO(b/130551176)
    ],
    deps = [
        ":combinations",
        ":distribute_lib",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras/layers",
    ],
)

cuda_py_test(
    name = "mirrored_variable_test",
    srcs = ["mirrored_variable_test.py"],
    tags = [
        "guitar",
        "multi_and_single_gpu",
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":distribute_lib",
        ":strategy_combinations",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras/layers",
    ],
)

distribute_py_test(
    name = "metrics_v1_test",
    srcs = ["metrics_v1_test.py"],
    main = "metrics_v1_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        ":tpu_strategy",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "keras_metrics_test",
    srcs = ["keras_metrics_test.py"],
    main = "keras_metrics_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        ":tpu_strategy",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "zero_batch_test",
    srcs = ["zero_batch_test.py"],
    main = "zero_batch_test.py",
    deps = [
        ":combinations",
        ":multi_worker_test_base",
        ":strategy_combinations",
        "//tensorflow/python:layers",
    ],
)

py_library(
    name = "model_collection_base",
    srcs = ["model_collection/model_collection_base.py"],
)

py_library(
    name = "simple_models",
    srcs = ["model_collection/simple_models.py"],
    deps = [
        ":model_collection_base",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/keras",
    ],
)

py_library(
    name = "saved_model_test_base",
    srcs = ["saved_model_test_base.py"],
    deps = [
        ":combinations",
        ":model_combinations",
        ":strategy_combinations",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model",
        "//third_party/py/numpy",
    ],
)

distribute_py_test(
    name = "saved_model_save_load_test",
    size = "medium",
    srcs = ["saved_model_save_load_test.py"],
    full_precision = True,
    main = "saved_model_save_load_test.py",
    shard_count = 5,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",
    ],
    deps = [
        ":saved_model_test_base",
        "//tensorflow/python/saved_model",
    ],
)

distribute_py_test(
    name = "keras_save_load_test",
    size = "medium",
    srcs = ["keras_save_load_test.py"],
    full_precision = True,
    main = "keras_save_load_test.py",
    shard_count = 7,
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":saved_model_test_base",
        "//tensorflow/python/keras/saving",
    ],
)

distribute_py_test(
    name = "saved_model_mixed_api_test",
    size = "medium",
    srcs = ["saved_model_mixed_api_test.py"],
    full_precision = True,
    main = "saved_model_mixed_api_test.py",
    shard_count = 5,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",
    ],
    deps = [
        ":saved_model_test_base",
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/saved_model",
    ],
)

distribute_py_test(
    name = "ctl_correctness_test",
    srcs = ["ctl_correctness_test.py"],
    main = "ctl_correctness_test.py",
    shard_count = 5,
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/python:keras_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:util",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:test",
    ],
)

cuda_py_test(
    name = "collective_all_reduce_strategy_test",
    srcs = ["collective_all_reduce_strategy_test.py"],
    tags = [
        "multi_and_single_gpu",
    ],
    # b/155301154 broken with XLA:GPU
    xla_enable_strict_auto_jit = True,
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":cross_device_utils",
        ":multi_worker_test_base",
        ":strategy_combinations",
        ":strategy_test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/mixed_precision/experimental:test_util",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "parameter_server_strategy_test",
    srcs = ["parameter_server_strategy_test.py"],
    tags = [
        "multi_and_single_gpu",
        "no_windows",  # TODO(b/156428279): reenable this test once the image is updated.
    ],
    # b/141096229: Non-atomic AssignAdd
    xla_enable_strict_auto_jit = False,
    deps = [
        ":central_storage_strategy",
        ":combinations",
        ":device_util",
        ":distribute_lib",
        ":multi_worker_test_base",
        ":multi_worker_util",
        ":parameter_server_strategy",
        ":ps_values",
        ":reduce_util",
        ":strategy_test_lib",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:training_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/estimator:run_config",
        "//tensorflow/python/keras/layers:core",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "multi_process_runner",
    srcs = ["multi_process_runner.py"],
    deps = [
        ":multi_process_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:tf2",
        "//tensorflow/python/compat:v2_compat",
        "@six_archive//:six",
    ],
)

py_library(
    name = "multi_process_lib",
    srcs = ["multi_process_lib.py"],
    deps = ["@six_archive//:six"],
)

py_test(
    name = "multi_process_runner_test",
    srcs = ["multi_process_runner_test.py"],
    python_version = "PY3",
    shard_count = 12,
    deps = [
        ":multi_process_runner",
        ":multi_worker_test_base",
        "//tensorflow/python/eager:test",
    ],
)

py_test(
    name = "multi_process_runner_no_init_test",
    srcs = ["multi_process_runner_no_init_test.py"],
    python_version = "PY3",
    deps = [
        ":multi_process_runner",
        ":multi_worker_test_base",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "distributed_file_utils",
    srcs = [
        "distributed_file_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":multi_worker_util",
    ],
)

py_test(
    name = "distributed_file_utils_test",
    srcs = ["distributed_file_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":distributed_file_utils",
        ":multi_worker_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "strategy_common_test",
    srcs = ["strategy_common_test.py"],
    tags = [
        "multi_and_single_gpu",
        # TODO(b/155301154): Enable this test on multi-gpu guitar once multi process
        # runner can run on guitar.
        "noguitar",
    ],
    xla_enable_strict_auto_jit = True,
    deps = [
        ":combinations",
        ":multi_worker_test_base",
        ":reduce_util",
        ":strategy_combinations",
        ":strategy_test_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "@absl_py//absl/testing:parameterized",
    ],
)
