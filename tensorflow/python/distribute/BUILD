load("//tensorflow:tensorflow.bzl", "py_test", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow/core/platform:default/distribute.bzl", "distribute_py_test")

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
    additional_deps = [
        ":all_reduce",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
    ],
)

py_library(
    name = "cross_device_ops",
    srcs = ["cross_device_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cross_device_utils",
        ":device_util",
        ":reduce_util",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:device_lib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:kernels",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/eager:context",
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
    additional_deps = [
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
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/ops/losses:loss_reduction",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_test(
    name = "distribute_lib_test",
    size = "small",
    srcs = ["distribute_lib_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = [
        "no_rocm",
    ],
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
        "//tensorflow/python:training_lib",
    ],
)

py_test(
    name = "distribute_coordinator_test",
    size = "medium",
    srcs = ["distribute_coordinator_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = ["no_oss_py2"],  # b/138443278
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
    name = "mirrored_strategy",
    srcs = ["mirrored_strategy.py"],
    deps = [
        ":cross_device_ops",
        ":device_util",
        ":distribute_lib",
        ":input_lib",
        ":multi_worker_util",
        ":numpy_dataset",
        ":reduce_util",
        ":shared_variable_creator",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
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
        ":input_lib",
        ":mirrored_strategy",
        ":numpy_dataset",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute:cross_device_ops",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:values",
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
        ":mirrored_strategy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/distribute:cross_device_ops",
        "//tensorflow/python/distribute:cross_device_utils",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/distribute:numpy_dataset",
        "//tensorflow/python/distribute:values",
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
    additional_deps = [
        ":distribute_lib",
        ":strategy_combinations",
        ":mirrored_function_strategy",
        ":values",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
    ],
    tags = ["no_pip"],
)

py_library(
    name = "multi_worker_util",
    srcs = [
        "multi_worker_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:training_lib",
    ],
)

cuda_py_test(
    name = "multi_worker_continuous_run_test",
    srcs = [
        "multi_worker_continuous_run_test.py",
    ],
    additional_deps = [
        ":collective_all_reduce_strategy",
        ":multi_process_runner",
        ":reduce_util",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:framework_ops",
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
    python_version = "PY2",
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
    additional_deps = [
        ":input_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python:errors",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "multi_worker_util_test",
    srcs = ["multi_worker_util_test.py"],
    python_version = "PY2",
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
        ":values",
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
    name = "shared_variable_creator",
    srcs = ["shared_variable_creator.py"],
)

py_test(
    name = "shared_variable_creator_test",
    srcs = ["shared_variable_creator_test.py"],
    python_version = "PY2",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/training/tracking:base",
        "@six_archive//:six",
    ],
)

py_library(
    name = "combinations",
    srcs = ["combinations.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python:framework_test_combinations_lib",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "strategy_combinations",
    srcs = ["strategy_combinations.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":central_storage_strategy",
        ":combinations",
        ":distribute_lib",
        ":mirrored_strategy",
        ":one_device_strategy",
        ":tpu_strategy",
        "//tensorflow/python:config",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras/optimizer_v2",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "strategy_combinations_test",
    srcs = ["strategy_combinations_test.py"],
    python_version = "PY2",
    deps = [
        ":combinations",
        ":reduce_util",
        ":strategy_combinations",
        "//tensorflow/python:config",
        "//tensorflow/python:constant_op",
        "//tensorflow/python/eager:test",
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
    additional_deps = [
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

distribute_py_test(
    name = "checkpointing_test",
    srcs = ["checkpointing_test.py"],
    main = "checkpointing_test.py",
    deps = [
        ":tpu_strategy",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
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
    deps = [
        ":collective_all_reduce_strategy",
        ":mirrored_strategy",
        ":reduce_util",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:values",
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
    additional_deps = [
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/distribute:cross_device_utils",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
)

cuda_py_test(
    name = "cross_device_ops_test",
    srcs = ["cross_device_ops_test.py"],
    additional_deps = [
        ":collective_all_reduce_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:multi_worker_test_base",
        ":mirrored_strategy",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/distribute:cross_device_ops",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
    tags = [
        # TODO(b/138143527): Re-enable after fixing Guitar failure.
        # "multi_and_single_gpu",
    ],
)

cuda_py_test(
    name = "one_device_strategy_test",
    srcs = ["one_device_strategy_test.py"],
    additional_deps = [
        ":strategy_test_lib",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/eager:test",
    ],
    grpc_enabled = True,
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
    additional_deps = [
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
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
    ],
)

distribute_py_test(
    name = "values_test",
    srcs = ["values_test.py"],
    main = "values_test.py",
    tags = [
        "no_oss",  # http://b/119349471
    ],
    deps = [
        ":mirrored_strategy",
        ":parameter_server_strategy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model/model_utils:mode_keys",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "moving_averages_test",
    srcs = ["moving_averages_test.py"],
    main = "moving_averages_test.py",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "custom_training_loop_test",
    srcs = ["custom_training_loop_test.py"],
    main = "custom_training_loop_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
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
        "no_rocm",
        "notap",  # TODO(b/139815303): enable after this is fixed.
    ],
    deps = [
        ":mirrored_strategy",
        ":single_loss_example",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
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
        "no_rocm",
    ],
    deps = [
        ":single_loss_example",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
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
    additional_deps = [
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

cuda_py_test(
    name = "remote_mirrored_strategy_eager_test",
    srcs = ["remote_mirrored_strategy_eager_test.py"],
    additional_deps = [
        ":combinations",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":strategy_test_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
)

cuda_py_test(
    name = "mirrored_strategy_test",
    srcs = ["mirrored_strategy_test.py"],
    additional_deps = [
        ":combinations",
        ":strategy_combinations",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":strategy_test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
    shard_count = 5,
    tags = [
        "guitar",
        "multi_and_single_gpu",
        "no_rocm",
        "no_windows_gpu",  # TODO(b/130551176)
    ],
)

cuda_py_test(
    name = "mirrored_variable_test",
    srcs = ["mirrored_variable_test.py"],
    additional_deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
    tags = [
        "guitar",
        "multi_and_single_gpu",
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
    shard_count = 5,
    deps = [
        ":saved_model_test_base",
        "//tensorflow/python/keras:saving",
    ],
)

distribute_py_test(
    name = "saved_model_mixed_api_test",
    size = "medium",
    srcs = ["saved_model_mixed_api_test.py"],
    full_precision = True,
    main = "saved_model_mixed_api_test.py",
    shard_count = 5,
    deps = [
        ":saved_model_test_base",
        "//tensorflow/python/keras:saving",
        "//tensorflow/python/saved_model",
    ],
)

distribute_py_test(
    name = "ctl_correctness_test",
    srcs = ["ctl_correctness_test.py"],
    main = "ctl_correctness_test.py",
    shard_count = 10,
    tags = [
        "multi_and_single_gpu",
        "noguitar",  # b/140755528
    ],
    deps = [
        "//tensorflow/python:keras_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:util",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/eager:test",
    ],
)

cuda_py_test(
    name = "collective_all_reduce_strategy_test",
    srcs = ["collective_all_reduce_strategy_test.py"],
    additional_deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":strategy_combinations",
        ":multi_worker_test_base",
        ":strategy_test_lib",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:cross_device_utils",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/keras/mixed_precision/experimental:test_util",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
    xla_enable_strict_auto_jit = False,
)

cuda_py_test(
    name = "parameter_server_strategy_test",
    srcs = ["parameter_server_strategy_test.py"],
    additional_deps = [
        ":parameter_server_strategy",
        ":central_storage_strategy",
        ":combinations",
        ":strategy_combinations",
        ":multi_worker_test_base",
        ":strategy_test_lib",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:layers",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/estimator:estimator_py",
    ],
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/133330625)
    ],
    # b/141096229: Non-atomic AssignAdd
    xla_enable_strict_auto_jit = False,
)

py_library(
    name = "multi_process_runner",
    srcs = ["multi_process_runner.py"],
    deps = [
        ":multi_process_lib",
        ":multi_worker_test_base",
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
        "//tensorflow/python/eager:test",
    ],
)

py_test(
    name = "multi_process_runner_no_init_test",
    srcs = ["multi_process_runner_no_init_test.py"],
    python_version = "PY3",
    deps = [
        ":multi_process_runner",
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
        "//tensorflow/python/distribute:multi_worker_util",
    ],
)
