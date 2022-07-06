load("//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "py_test", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow/core/platform/default:distribute.bzl", "distribute_py_test")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")

package(
    default_visibility = [
        "//tensorflow:internal",
        "//third_party/py/keras:__subpackages__",  # TODO(scottzhu): remove this once keras is relying on tf.__internal__.
    ],
    licenses = ["notice"],
)

py_library(
    name = "distribute_test_lib_pip",
    srcs_version = "PY3",
    deps = [
        ":combinations",
        ":multi_worker_test_base",
        ":single_loss_example",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        ":values_v2",
        "//tensorflow/python/distribute/v1:all_reduce",
    ],
)

py_library(
    name = "cross_device_ops",
    srcs = ["cross_device_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":collective_util",
        ":cross_device_utils",
        ":device_util",
        ":distribute_utils",
        ":ps_values",
        ":reduce_util",
        ":tpu_values",
        ":values",
        ":values_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:kernels",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/client:device_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
        "@six_archive//:six",
    ],
)

pytype_strict_library(
    name = "cross_device_utils",
    srcs = ["cross_device_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":collective_util",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nccl_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/types",
    ],
)

py_library(
    name = "device_util",
    srcs = ["device_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:device",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "device_util_test",
    srcs = ["device_util_test.py"],
    python_version = "PY3",
    tags = ["no_oss_py2"],
    deps = [
        ":combinations",
        ":device_util",
        ":multi_worker_test_base",
        ":multi_worker_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python/eager:context",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "distribute",
    srcs_version = "PY3",
    deps = [
        ":cross_device_ops",
        ":distribute_lib",
        ":merge_call_interim",
        ":mirrored_strategy",
        ":multi_process_runner",
        ":multi_worker_test_base",
        ":one_device_strategy",
        ":parameter_server_strategy_v2",
        ":sharded_variable",
        "//tensorflow/python/distribute/coordinator:cluster_coordinator",
        "//tensorflow/python/distribute/experimental",
        "//tensorflow/python/distribute/failure_handling:failure_handling_lib",
    ],
)

py_library(
    name = "distribute_lib",
    srcs = [
        "distribute_lib.py",
        "distribution_strategy_context.py",
    ],
    srcs_version = "PY3",
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
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/ops/losses",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_test(
    name = "distribute_lib_test",
    size = "small",
    srcs = ["distribute_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
    ],
)

py_library(
    name = "distribute_config",
    srcs = [
        "distribute_config.py",
    ],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "distribute_coordinator",
    srcs = [
        "distribute_coordinator.py",
    ],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    tags = [
        "notsan",  # TODO(b/220133218)
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
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "mirrored_run",
    srcs = ["mirrored_run.py"],
    srcs_version = "PY3",
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
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/impl",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "distribute_utils",
    srcs = ["distribute_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":device_util",
        ":distribute_lib",
        ":ps_values",
        ":reduce_util",
        ":sharded_variable",
        ":shared_variable_creator",
        ":tpu_values",
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
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/impl",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "tpu_util",
    srcs = ["tpu_util.py"],
    deps = [
        ":packed_distributed_variable",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/tpu:tpu_py",
    ],
)

py_library(
    name = "mirrored_strategy",
    srcs = ["mirrored_strategy.py"],
    srcs_version = "PY3",
    deps = [
        ":collective_util",
        ":cross_device_ops",
        ":device_util",
        ":distribute_lib",
        ":distribute_utils",
        ":input_lib",
        ":input_util",
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
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
    ],
)

py_library(
    name = "parameter_server_strategy",
    srcs = ["parameter_server_strategy.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":cross_device_ops",
        ":device_util",
        ":distribute_lib",
        ":distribute_utils",
        ":input_lib",
        ":input_util",
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
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "central_storage_strategy",
    srcs = ["central_storage_strategy.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":distribute_lib",
        ":input_lib",
        ":input_util",
        ":numpy_dataset",
        ":reduce_util",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

py_library(
    name = "collective_all_reduce_strategy",
    srcs = ["collective_all_reduce_strategy.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":collective_util",
        ":cross_device_ops",
        ":cross_device_utils",
        ":device_util",
        ":distribute_lib",
        ":distribute_utils",
        ":input_lib",
        ":input_util",
        ":mirrored_strategy",
        ":multi_worker_util",
        ":numpy_dataset",
        ":reduce_util",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/tpu:tpu_strategy_util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "multi_worker_util",
    srcs = [
        "multi_worker_util.py",
    ],
    srcs_version = "PY3",
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
    python_version = "PY3",
    tags = [
        "no_windows",  # TODO(b/184424727): Re-enable this.
        "noasan",  # TODO(b/180630068)
        "nomsan",  # TODO(b/180630068)
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        ":device_util",
        ":distribute_lib",
        ":distribute_utils",
        ":input_ops",
        ":values",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/experimental/ops:cardinality",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:multi_device_iterator_ops",
        "//tensorflow/python/data/ops:optional_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_library(
    name = "input_ops",
    srcs = ["input_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/util:nest",
    ],
)

cuda_py_test(
    name = "input_ops_test",
    srcs = ["input_ops_test.py"],
    python_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":cross_device_ops",
        ":device_util",
        ":distribute_lib",
        ":distribute_utils",
        ":input_lib",
        ":input_util",
        ":numpy_dataset",
        ":reduce_util",
        ":tpu_replicated_variable",
        ":tpu_util",
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
        "//tensorflow/python:variables",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/impl",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/tpu:device_assignment",
        "//tensorflow/python/tpu:tpu_hardware_feature",
        "//tensorflow/python/tpu:tpu_lib",
        "//tensorflow/python/tpu:tpu_py",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
    ],
)

distribute_py_test(
    name = "random_generator_test",
    srcs = ["random_generator_test.py"],
    main = "random_generator_test.py",
    shard_count = 12,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # b/213388775
    ],
    tpu_tags = [
        "no_oss",
    ],
    xla_enable_strict_auto_jit = False,  # PSStrategy doesn't work on _xla tests
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:stateful_random_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
    ],
)

tpu_py_test(
    name = "tpu_strategy_test",
    srcs = ["tpu_strategy_test.py"],
    disable_experimental = True,
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":strategy_test_lib",
        ":tpu_strategy",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_test(
    name = "tpu_strategy_compilation_test",
    srcs = ["tpu_strategy_compilation_test.py"],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":tpu_strategy",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
    ],
)

# Used only by estimator.
py_library(
    name = "estimator_training",
    srcs = [
        "estimator_training.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":distribute_coordinator",
        ":distribute_coordinator_context",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "reduce_util",
    srcs = ["reduce_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "collective_util",
    srcs = ["collective_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
    ],
)

tf_py_test(
    name = "collective_util_test",
    srcs = ["collective_util_test.py"],
    deps = [
        ":collective_util",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "shared_variable_creator",
    srcs = ["shared_variable_creator.py"],
    srcs_version = "PY3",
)

py_test(
    name = "shared_variable_creator_test",
    srcs = ["shared_variable_creator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        ":distribute_lib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_util",
    ],
)

py_library(
    name = "packed_distributed_variable",
    srcs = ["packed_distributed_variable.py"],
    srcs_version = "PY3",
    deps = [
        ":device_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "values",
    srcs = ["values.py"],
    srcs_version = "PY3",
    deps = [
        ":device_util",
        ":distribute_lib",
        ":packed_distributed_variable",
        ":reduce_util",
        ":values_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/types",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "values_v2",
    srcs = ["values_v2.py"],
    deps = [
        ":device_util",
        ":distribute_lib",
        ":reduce_util",
        ":tpu_util",
        ":values",
        ":values_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/types",
    ],
)

distribute_py_test(
    name = "values_v2_test",
    srcs = ["values_v2_test.py"],
    tags = [
        "no_mac",  # b/190644499
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        ":test_util",
        ":values_v2",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "ps_values",
    srcs = ["ps_values.py"],
    srcs_version = "PY3",
    deps = [
        ":distribute_lib",
        ":values",
        ":values_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute/coordinator:coordinator_context",
        "//tensorflow/python/keras/saving/saved_model:load_context",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types",
    ],
)

py_library(
    name = "values_util",
    srcs = ["values_util.py"],
    srcs_version = "PY3",
    deps = [
        ":distribute_lib",
        ":reduce_util",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model:save_options",
    ],
)

py_library(
    name = "tpu_values",
    srcs = ["tpu_values.py"],
    srcs_version = "PY3",
    deps = [
        ":packed_distributed_variable",
        ":tpu_replicated_variable",
        ":tpu_util",
        ":values",
        ":values_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops_gen",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
    ],
)

py_library(
    name = "tpu_replicated_variable",
    srcs = ["tpu_replicated_variable.py"],
    srcs_version = "PY3",
    deps = [
        ":tpu_util",
        "//tensorflow/compiler/xla/experimental/xla_sharding",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:resource_variable_ops_gen",
        "//tensorflow/python:tpu_partition_ops_gen",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/trackable:base",
    ],
)

tpu_py_test(
    name = "tpu_replicated_variable_test",
    srcs = ["tpu_replicated_variable_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_replicated_variable",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "combinations",
    srcs = ["combinations.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
        "//third_party/py/keras:__subpackages__",
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":distribute_lib",
        ":multi_process_runner",
        ":multi_worker_test_base",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_combinations_lib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:tf_decorator",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

distribute_py_test(
    name = "combinations_test",
    srcs = ["combinations_test.py"],
    python_version = "PY3",
    tags = ["multi_gpu"],
    deps = [
        ":combinations",
        ":test_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python/distribute/cluster_resolver:tfconfig_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "strategy_combinations",
    srcs = ["strategy_combinations.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
        "//third_party/py/keras:__subpackages__",
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
        ":parameter_server_strategy_v2",
        ":sharded_variable",
        ":test_util",
        ":tpu_strategy",
        "//tensorflow/python:platform",
        "//tensorflow/python:tf2",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/tpu:device_assignment",
        "//tensorflow/python/tpu:tpu_lib",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:tf_export",
    ],
)

distribute_py_test(
    name = "strategy_combinations_test",
    srcs = ["strategy_combinations_test.py"],
    disable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_cuda_asan",
        "noasan",
    ],  # TODO(b/195246941) b/196591124
    deps = [
        ":central_storage_strategy",
        ":collective_all_reduce_strategy",
        ":combinations",
        ":mirrored_strategy",
        ":one_device_strategy",
        ":parameter_server_strategy_v2",
        ":reduce_util",
        ":strategy_combinations",
        ":test_util",
        ":tpu_strategy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:def_function",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "multi_worker_test_base",
    srcs = ["multi_worker_test_base.py"],
    srcs_version = "PY3",
    deps = [
        ":distribute_coordinator",
        ":multi_process_runner",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:remote",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "multi_worker_test_base_test",
    srcs = ["multi_worker_test_base_test.py"],
    srcs_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/170834611)
    ],
    deps = [
        ":multi_worker_test_base",
    ],
)

cuda_py_test(
    name = "checkpoint_utils_test",
    size = "medium",
    srcs = ["checkpoint_utils_test.py"],
    python_version = "PY3",
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
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/eager:test",
    ],
)

distribute_py_test(
    name = "input_lib_test",
    srcs = ["input_lib_test.py"],
    disable_mlir_bridge = False,
    main = "input_lib_test.py",
    shard_count = 10,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # TODO(b/214574707): times out
        "notsan",  # TODO(b/177098062): flaky for over a year
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":input_lib",
        ":input_util",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":reduce_util",
        ":strategy_combinations",
        ":test_util",
        ":values",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/v1:input_lib",
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
        ":tpu_strategy",
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
    python_version = "PY3",
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
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":collective_util",
        ":combinations",
        ":cross_device_ops",
        ":cross_device_utils",
        ":multi_process_runner",
        ":multi_worker_test_base",
        ":reduce_util",
        ":test_util",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:indexed_slices",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "one_device_strategy_test",
    srcs = ["one_device_strategy_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/saved_model:revived_types",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sharded_variable_test",
    size = "small",
    srcs = ["sharded_variable_test.py"],
    deps = [
        ":combinations",
        ":multi_worker_test_base",
        ":parameter_server_strategy_v2",
        ":sharded_variable",
        ":test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/distribute/coordinator:cluster_coordinator",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/module",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/training:server_lib",
    ],
)

py_library(
    name = "strategy_test_lib",
    srcs = ["strategy_test_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":collective_all_reduce_strategy",
        ":distribute_lib",
        ":distribute_utils",
        ":mirrored_strategy",
        ":reduce_util",
        ":tpu_strategy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/platform",
        "//tensorflow/python/training:training_util",
        "//third_party/py/numpy",
    ],
)

distribute_py_test(
    name = "values_test",
    size = "medium",
    srcs = ["values_test.py"],
    main = "values_test.py",
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "no_oss",  # b/178656226
        "noasan",  # b/175816710
        "notsan",  # b/168645872
    ],
    tpu_tags = [
        "noasan",  # TODO(b/337374867) fails with -fsanitize=null
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        ":test_util",
        ":tpu_strategy",
        ":tpu_values",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tf2",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "distributed_variable_test",
    size = "medium",
    srcs = ["distributed_variable_test.py"],
    disable_v2 = True,  # TODO(b/209058825)
    disable_v3 = True,  # TODO(b/209058825)
    main = "distributed_variable_test.py",
    shard_count = 3,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "no_oss",  # b/178656226
        "noasan",  # b/175816710
        "notap",  # Flaky
        "notsan",  # b/168645872
    ],
    tpu_tags = [
        "noasan",  # TODO(b/337374867) fails with -fsanitize=null
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":distribute_lib",
        ":distribute_utils",
        ":packed_distributed_variable",
        ":parameter_server_strategy",
        ":ps_values",
        ":strategy_combinations",
        ":test_util",
        ":tpu_strategy",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:indexed_slices",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/types",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "mirrored_values_test",
    size = "medium",
    srcs = ["mirrored_values_test.py"],
    main = "mirrored_values_test.py",
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "noasan",  # b/175816710
        "notsan",  # b/168645872
    ],
    tpu_tags = [
        "noasan",  # TODO(b/337374867) fails with -fsanitize=null
    ],
    deps = [
        ":combinations",
        ":distribute_lib",
        ":distribute_utils",
        ":packed_distributed_variable",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        ":tpu_strategy",
        ":tpu_values",
        ":values",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/types",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "per_replica_test",
    size = "medium",
    srcs = ["per_replica_test.py"],
    main = "per_replica_test.py",
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "noasan",  # b/175816710
        "notsan",  # b/168645872
    ],
    tpu_tags = [
        "noasan",  # TODO(b/337374867) fails with -fsanitize=null
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":distribute_lib",
        ":distribute_utils",
        ":packed_distributed_variable",
        ":parameter_server_strategy",
        ":ps_values",
        ":strategy_combinations",
        ":test_util",
        ":tpu_strategy",
        ":tpu_values",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tf2",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/types",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "distribute_utils_test",
    srcs = ["distribute_utils_test.py"],
    disable_mlir_bridge = False,
    main = "distribute_utils_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":distribute_utils",
        ":strategy_combinations",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model/model_utils:mode_keys",
        "@absl_py//absl/testing:parameterized",
        "@wrapt",
    ],
)

distribute_py_test(
    name = "vars_test",
    size = "medium",
    srcs = ["vars_test.py"],
    main = "vars_test.py",
    shard_count = 5,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":combinations",
        ":distribute_lib",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:indexed_slices",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/tpu:tpu_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "ps_values_test",
    size = "medium",
    srcs = ["ps_values_test.py"],
    disable_mlir_bridge = False,
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
    shard_count = 5,
    tags = [
        "multi_gpu",
        "no_windows",  # TODO(b/184424727): Re-enable this.
        "notpu",  # TODO(b/210145904)
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "custom_training_loop_gradient_test",
    srcs = ["custom_training_loop_gradient_test.py"],
    disable_mlir_bridge = False,
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
        "no_oss",  # TODO(b/227211015)
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
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "single_loss_example",
    srcs = ["single_loss_example.py"],
    srcs_version = "PY3",
    deps = [
        ":step_fn",
        ":strategy_test_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "step_fn",
    srcs = ["step_fn.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:training",
        "//tensorflow/python/eager:backprop",
    ],
)

cuda_py_test(
    name = "warm_starting_util_test",
    size = "medium",
    srcs = ["warm_starting_util_test.py"],
    python_version = "PY3",
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
    python_version = "PY3",
    tags = [
        "no_windows",  # TODO(b/197981388): Re-enable this.
    ],
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
    python_version = "PY3",
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
    ],
)

cuda_py_test(
    name = "mirrored_variable_test",
    srcs = ["mirrored_variable_test.py"],
    python_version = "PY3",
    tags = [
        "guitar",
        "multi_and_single_gpu",
        "no_windows",  # TODO(b/184424727): Re-enable this.
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":distribute_lib",
        ":distribute_utils",
        ":strategy_combinations",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:config",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
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
        ":strategy_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "zero_batch_test",
    srcs = ["zero_batch_test.py"],
    disable_mlir_bridge = False,
    main = "zero_batch_test.py",
    tags = [
        "no_oss",  # Keras is not available in OSS test
    ],
    deps = [
        ":combinations",
        ":multi_worker_test_base",
        ":strategy_combinations",
    ],
)

cuda_py_test(
    name = "collective_all_reduce_strategy_test",
    srcs = ["collective_all_reduce_strategy_test.py"],
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "nomsan",  # b/154224457: Re-enable when fixed.
        "notsan",  # TODO(b/220133218)
    ],
    # b/155301154 broken with XLA:GPU
    xla_enable_strict_auto_jit = True,
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":cross_device_utils",
        ":distribute_lib",
        ":distribute_utils",
        ":input_lib",
        ":multi_worker_test_base",
        ":multi_worker_util",
        ":reduce_util",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:device",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/tpu:tpu_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_test(
    name = "collective_all_reduce_strategy_test_tpu",
    srcs = ["collective_all_reduce_strategy_test.py"],
    # FIXME(b/227404010): On TFRT TPU, eager CollectiveReduceV2 is broken.
    disable_tfrt = True,
    main = "collective_all_reduce_strategy_test.py",
    python_version = "PY3",
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":cross_device_utils",
        ":distribute_lib",
        ":distribute_utils",
        ":input_lib",
        ":multi_worker_test_base",
        ":multi_worker_util",
        ":reduce_util",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:device",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/tpu:tpu_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "parameter_server_strategy_test",
    srcs = ["parameter_server_strategy_test.py"],
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "notsan",  # TODO(b/220133218)
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
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "multi_process_runner",
    srcs = ["multi_process_runner.py"],
    srcs_version = "PY3",
    deps = [
        ":multi_process_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:tf2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:context",
        "@absl_py//absl/logging",
        "@dill_archive//:dill",
        "@six_archive//:six",
        "@tblib_archive//:tblib",
    ],
)

py_library(
    name = "multi_process_lib",
    srcs = ["multi_process_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/eager:test",
        "@absl_py//absl:app",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "packed_distributed_variable_test",
    srcs = ["packed_distributed_variable_test.py"],
    tags = [
        "nomac",  #TODO(b/145922293): It would cause a Python segfault on macos
    ],
    deps = [
        ":device_util",
        ":packed_distributed_variable",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
    ],
)

cuda_py_test(
    name = "multi_process_runner_test",
    srcs = ["multi_process_runner_test.py"],
    python_version = "PY3",
    shard_count = 12,
    tags = [
        "multi_gpu",
        "noasan",
        "nomsan",
    ],  # b/175904958
    deps = [
        ":combinations",
        ":multi_process_runner",
        ":multi_worker_test_base",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:parameterized",
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

distribute_py_test(
    name = "strategy_common_test",
    srcs = ["strategy_common_test.py"],
    disable_mlir_bridge = False,
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "multi_and_single_gpu",
        "notsan",  # TODO(b/160006974)
    ],
    xla_enable_strict_auto_jit = True,
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":distribute_lib",
        ":multi_worker_test_base",
        ":reduce_util",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        ":tpu_strategy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "strategy_gather_test",
    srcs = ["strategy_gather_test.py"],
    disable_mlir_bridge = False,
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "notsan",  # TODO(b/160006974)
    ],
    xla_enable_strict_auto_jit = True,
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":multi_worker_test_base",
        ":reduce_util",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "tf_function_test",
    srcs = ["tf_function_test.py"],
    disable_mlir_bridge = False,
    main = "tf_function_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":device_util",
        ":strategy_combinations",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model:save_options",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY3",
    deps = [
        ":collective_all_reduce_strategy",
        ":multi_process_runner",
        ":tpu_strategy",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:config",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:context",
        "@absl_py//absl:app",
    ],
)

distribute_py_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    disable_mlir_bridge = False,
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        ":test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "parameter_server_strategy_v2",
    srcs = ["parameter_server_strategy_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":distribute_lib",
        ":distribute_utils",
        ":input_util",
        ":parameter_server_strategy",
        ":sharded_variable",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tf_decorator",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training:server_lib",
    ],
)

distribute_py_test(
    name = "parameter_server_strategy_v2_test",
    srcs = ["parameter_server_strategy_v2_test.py"],
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "nomac",  # TODO(b/201788023): Attempt MultiProcessCluster to fix this.
        "notpu",
        "notsan",  # Tsan failure doesn't seem to be caused by TF.
    ],
    deps = [
        ":combinations",
        ":multi_worker_test_base",
        ":parameter_server_strategy_v2",
        ":sharded_variable",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:linalg_ops_impl",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "distributed_table_test",
    srcs = ["distributed_table_test.py"],
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "notpu",
        "notsan",  # Tsan failure doesn't seem to be caused by TF.
    ],
    deps = [
        ":combinations",
        ":device_util",
        ":multi_worker_test_base",
        ":parameter_server_strategy_v2",
        ":ps_values",
        ":test_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/distribute/coordinator:cluster_coordinator",
        "//tensorflow/python/distribute/coordinator:coordinator_context",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/module",
        "//tensorflow/python/platform",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_test(
    name = "tpu_strategy_model_parallelism_test",
    srcs = ["tpu_strategy_model_parallelism_test.py"],
    disable_experimental = True,  # b/202779350
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":distribute_lib",
        ":strategy_test_lib",
        ":tpu_strategy",
        ":tpu_values",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/module",
        "//tensorflow/python/platform",
        "//tensorflow/python/tpu:device_assignment",
        "//tensorflow/python/tpu:tpu_lib",
        "//tensorflow/python/training:checkpoint_management",
    ],
)

pytype_strict_library(
    name = "input_util",
    srcs = ["input_util.py"],
    srcs_version = "PY3",
    deps = [
        ":input_lib",
        "//tensorflow/python:tf2",
        "//tensorflow/python/distribute/v1:input_lib",
    ],
)

py_library(
    name = "merge_call_interim",
    srcs = [
        "merge_call_interim.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":distribute_lib",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_test(
    name = "template_mirrored_strategy_test",
    size = "small",
    srcs = ["template_mirrored_strategy_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:mirrored_strategy",
    ],
)
