load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")
load("//tensorflow/core/platform:distribute.bzl", "distribute_py_strict_test")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "distribute_test_lib_pip",
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

py_strict_library(
    name = "cross_device_ops",
    srcs = ["cross_device_ops.py"],
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
        "//tensorflow/python/client:device_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:kernels",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

pytype_strict_library(
    name = "cross_device_utils",
    srcs = ["cross_device_utils.py"],
    deps = [
        ":collective_util",
        ":values",
        "//tensorflow/python/eager:backprop_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:collective_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nccl_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/types:core",
    ],
)

py_strict_library(
    name = "device_util",
    srcs = ["device_util.py"],
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:ops",
    ],
)

cuda_py_strict_test(
    name = "device_util_test",
    srcs = ["device_util_test.py"],
    deps = [
        ":combinations",
        ":device_util",
        ":multi_worker_test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "distribute",
    srcs = ["__init__.py"],
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
        "//tensorflow/python/distribute/failure_handling:preemption_watcher",
    ],
)

py_strict_library(
    name = "distribute_lib",
    srcs = [
        "distribute_lib.py",
    ],
    deps = [
        ":collective_util",
        ":device_util",
        ":numpy_dataset",
        ":reduce_util",
        "//tensorflow/python:tf2",
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/autograph/impl:api",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:ref_variable",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types:distribute",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
        "@six_archive//:six",
    ],
)

py_strict_test(
    name = "distribute_lib_test",
    size = "small",
    srcs = ["distribute_lib_test.py"],
    deps = [
        ":combinations",
        ":distribute_lib",
        ":input_lib",
        ":reduce_util",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "distribute_config",
    srcs = [
        "distribute_config.py",
    ],
    deps = [],
)

py_strict_library(
    name = "distribute_coordinator",
    srcs = [
        "distribute_coordinator.py",
    ],
    deps = [
        ":distribute_coordinator_context",
        ":multi_worker_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:coordinator",
        "//tensorflow/python/training:monitored_session",
        "//tensorflow/python/training:server_lib",
    ],
)

py_strict_test(
    name = "distribute_coordinator_test",
    size = "medium",
    srcs = ["distribute_coordinator_test.py"],
    tags = [
        "notsan",  # TODO(b/220133218)
    ],
    deps = [
        ":distribute_coordinator",
        ":distribute_coordinator_context",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:coordinator",
        "//tensorflow/python/training:monitored_session",
        "//tensorflow/python/training:session_manager",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "distribute_coordinator_context",
    srcs = [
        "distribute_coordinator_context.py",
    ],
    deps = [],
)

py_strict_library(
    name = "mirrored_run",
    srcs = ["mirrored_run.py"],
    deps = [
        ":distribute_lib",
        ":distribute_utils",
        ":shared_variable_creator",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/autograph/impl:api",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:coordinator",
        "//tensorflow/python/util:traceback_utils",
    ],
)

py_strict_library(
    name = "distribute_utils",
    srcs = ["distribute_utils.py"],
    deps = [
        ":distribute_lib",
        ":reduce_util",
        ":tpu_values",
        ":values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops/losses:losses_impl",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "tpu_util",
    srcs = ["tpu_util.py"],
    deps = [
        ":packed_distributed_variable",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/tpu:tpu_replication",
    ],
)

py_strict_library(
    name = "mirrored_strategy",
    srcs = ["mirrored_strategy.py"],
    deps = [
        ":collective_util",
        ":cross_device_ops",
        ":cross_device_utils",
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
        ":values_util",
        "//tensorflow/python:tf2",
        "//tensorflow/python/distribute/cluster_resolver:tfconfig_cluster_resolver_py",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "parameter_server_strategy",
    srcs = ["parameter_server_strategy.py"],
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
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/distribute/cluster_resolver:tfconfig_cluster_resolver_py",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:device_setter",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "central_storage_strategy",
    srcs = ["central_storage_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":device_util",
        ":distribute_lib",
        ":parameter_server_strategy",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "one_device_strategy",
    srcs = ["one_device_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":device_util",
        ":distribute_lib",
        ":distribute_utils",
        ":input_lib",
        ":input_util",
        ":numpy_dataset",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "collective_all_reduce_strategy",
    srcs = ["collective_all_reduce_strategy.py"],
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
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/distribute/cluster_resolver:tfconfig_cluster_resolver_py",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:collective_ops",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "multi_worker_util",
    srcs = [
        "multi_worker_util.py",
    ],
    deps = [
        ":distribute_coordinator_context",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/training:server_lib",
    ],
)

cuda_py_strict_test(
    name = "multi_worker_continuous_run_test",
    srcs = [
        "multi_worker_continuous_run_test.py",
    ],
    tags = [
        "no_pip",  # TODO(b/266520226)
        "no_windows",  # TODO(b/184424727): Re-enable this.
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":multi_process_runner",
        ":multi_worker_test_base",
        ":reduce_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:variable_scope",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@pypi_dill//:pkg",  # build_cleaner: keep
    ],
)

py_strict_library(
    name = "numpy_dataset",
    srcs = ["numpy_dataset.py"],
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "numpy_dataset_test",
    size = "small",
    srcs = ["numpy_dataset_test.py"],
    deps = [
        ":numpy_dataset",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:variable_v1",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "input_lib",
    srcs = ["input_lib.py"],
    deps = [
        ":device_util",
        ":distribute_lib",
        ":distribute_utils",
        ":input_ops",
        ":reduce_util",
        ":values",
        "//tensorflow/python/autograph/operators:py_builtins",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/experimental/ops:cardinality",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:multi_device_iterator_ops",
        "//tensorflow/python/data/ops:optional_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/types:distribute",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "input_ops",
    srcs = ["input_ops.py"],
    deps = [
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/util:traverse",
        "//tensorflow/python/framework:op_def_registry",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/types:data",
        "//tensorflow/python/types:distribute",
    ],
)

cuda_py_strict_test(
    name = "input_ops_test",
    srcs = ["input_ops_test.py"],
    deps = [
        ":input_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:python_io",
        "//tensorflow/python/ops:dataset_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_test(
    name = "multi_worker_util_test",
    srcs = ["multi_worker_util_test.py"],
    deps = [
        ":multi_worker_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/training:server_lib",
    ],
)

cuda_py_strict_test(
    name = "mwms_pjrt_gpu_test",
    srcs = ["mwms_pjrt_gpu_test.py"],
    tags = [
        "multi_and_single_gpu",
    ],
    xla_enabled = True,
    deps = [
        ":multi_process_runner",
        ":multi_worker_test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:collective_ops",
    ],
)

py_strict_library(
    name = "tpu_strategy",
    srcs = ["tpu_strategy.py"],
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
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/autograph/impl:api",
        "//tensorflow/python/compiler/xla/experimental:xla_sharding",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:device_spec",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/tpu:device_assignment",
        "//tensorflow/python/tpu:tpu_hardware_feature",
        "//tensorflow/python/tpu:tpu_lib",
        "//tensorflow/python/tpu:tpu_py",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
    ],
)

distribute_py_strict_test(
    name = "random_generator_test",
    srcs = ["random_generator_test.py"],
    env = {
        # multi-process test uses too much memory otherwise. b/289970206
        "CUDA_MODULE_LOADING": "LAZY",
    },
    main = "random_generator_test.py",
    shard_count = 12,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # b/213388775
        "no_oss",  # b/241013307
    ],
    tpu_tags = [
        "no_oss",
    ],
    xla_enable_strict_auto_jit = False,  # PSStrategy doesn't work on _xla tests
    deps = [
        ":combinations",
        ":mirrored_strategy",
        ":multi_process_runner",
        ":sharded_variable",
        ":strategy_combinations",
        ":values",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute/coordinator:cluster_coordinator",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_combinations_lib",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:stateful_random_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/util:deprecation",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_strategy_test",
    srcs = ["tpu_strategy_test.py"],
    args = ["--tpu_use_tfrt=true"],
    disable_experimental = True,
    tags = ["no_oss"],
    deps = [
        ":distribute_lib",
        ":packed_distributed_variable",
        ":reduce_util",
        ":strategy_test_lib",
        ":tpu_strategy",
        ":tpu_values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_switch_case",
        "//tensorflow/python/ops:dataset_ops_gen",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:logging_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/tpu:device_assignment",
        "//tensorflow/python/tpu:tpu_hardware_feature",
        "//tensorflow/python/tpu:tpu_replication",
        "//tensorflow/python/tpu:tpu_strategy_util",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_strategy_compilation_test",
    srcs = ["tpu_strategy_compilation_test.py"],
    disable_experimental = True,
    disable_mlir_bridge = False,
    tags = ["no_oss"],
    deps = [
        ":tpu_strategy",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:flags",
    ],
)

py_strict_library(
    name = "reduce_util",
    srcs = ["reduce_util.py"],
    deps = [
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "collective_util",
    srcs = ["collective_util.py"],
    deps = [
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "collective_util_test",
    srcs = ["collective_util_test.py"],
    deps = [
        ":collective_util",
        "//tensorflow/python/eager:test",
    ],
)

py_strict_library(
    name = "shared_variable_creator",
    srcs = ["shared_variable_creator.py"],
)

py_strict_test(
    name = "shared_variable_creator_test",
    srcs = ["shared_variable_creator_test.py"],
    deps = [
        ":shared_variable_creator",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
    ],
)

py_strict_library(
    name = "summary_op_util",
    srcs = ["summary_op_util.py"],
    deps = [
        ":distribute_lib",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_strict_library(
    name = "packed_distributed_variable",
    srcs = ["packed_distributed_variable.py"],
    deps = [
        ":device_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
    ],
)

py_strict_library(
    name = "values",
    srcs = ["values.py"],
    deps = [
        ":device_util",
        ":distribute_lib",
        ":packed_distributed_variable",
        ":reduce_util",
        ":values_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/types:core",
        "//tensorflow/python/types:distribute",
        "//tensorflow/python/types:trace",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "values_v2",
    srcs = ["values_v2.py"],
    deps = [
        ":device_util",
        ":tpu_util",
        ":values_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
    ],
)

distribute_py_strict_test(
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
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "ps_values",
    srcs = ["ps_values.py"],
    deps = [
        ":distribute_lib",
        ":distribute_utils",
        ":values",
        ":values_util",
        "//tensorflow/python/distribute/coordinator:coordinator_context",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:handle_data_util",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:numpy_compat",
    ],
)

py_strict_library(
    name = "values_util",
    srcs = ["values_util.py"],
    deps = [
        ":distribute_lib",
        ":reduce_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/training/saving:saveable_object",
    ],
)

py_strict_library(
    name = "tpu_values",
    srcs = ["tpu_values.py"],
    visibility = [
        "//smartass/brain/ops:__subpackages__",
        "//tensorflow:internal",
    ],
    deps = [
        ":packed_distributed_variable",
        ":tpu_replicated_variable",
        ":tpu_util",
        ":values",
        ":values_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops_gen",
        "//tensorflow/python/ops:variable_scope",
    ],
)

py_strict_library(
    name = "tpu_replicated_variable",
    srcs = ["tpu_replicated_variable.py"],
    deps = [
        ":tpu_util",
        "//tensorflow/python/compiler/xla/experimental:xla_sharding",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:resource_variable_ops_gen",
        "//tensorflow/python/ops:tpu_partition_ops_gen",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/trackable:base",
    ],
)

tpu_py_strict_test(
    name = "tpu_replicated_variable_test",
    srcs = ["tpu_replicated_variable_test.py"],
    deps = [
        ":tpu_replicated_variable",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:variables",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "combinations",
    srcs = ["combinations.py"],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":distribute_lib",
        ":multi_process_runner",
        ":multi_worker_test_base",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_combinations_lib",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "@absl_py//absl:app",
        "@six_archive//:six",
    ],
)

distribute_py_strict_test(
    name = "combinations_test",
    srcs = ["combinations_test.py"],
    tags = ["multi_gpu"],
    deps = [
        ":combinations",
        ":test_util",
        "//tensorflow/python/distribute/cluster_resolver:tfconfig_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "strategy_combinations",
    srcs = ["strategy_combinations.py"],
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
        ":parameter_server_strategy_v2",
        ":sharded_variable",
        ":test_util",
        ":tpu_strategy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/tpu:device_assignment",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:tf_export",
    ],
)

distribute_py_strict_test(
    name = "strategy_combinations_test",
    srcs = ["strategy_combinations_test.py"],
    disable_mlir_bridge = False,
    tags = [
        "no_cuda_asan",  # TODO(b/195246941) b/196591124
        "no_gpu",  # b/276511061
    ],
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
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "multi_worker_test_base",
    srcs = ["multi_worker_test_base.py"],
    visibility = [
        "//smartass/brain/ops:__subpackages__",
        "//tensorflow:internal",
    ],
    deps = [
        ":distribute_coordinator",
        ":multi_process_runner",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/distribute/cluster_resolver:tfconfig_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:coordinator",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

tf_py_strict_test(
    name = "multi_worker_test_base_test",
    srcs = ["multi_worker_test_base_test.py"],
    tags = [
        "no_oss",  # TODO(b/170834611)
    ],
    deps = [
        ":multi_process_runner",
        ":multi_worker_test_base",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
    ],
)

cuda_py_strict_test(
    name = "checkpoint_utils_test",
    size = "medium",
    srcs = ["checkpoint_utils_test.py"],
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:checkpoint_utils",
        "//tensorflow/python/training:saver",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
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
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:variables",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
    name = "input_lib_test",
    srcs = ["input_lib_test.py"],
    disable_mlir_bridge = False,
    main = "input_lib_test.py",
    shard_count = 10,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # TODO(b/214574707): times out
        "no_oss",  # TODO(b/284980046): Breaking nightlies, fails when unable to start GRPC server.
        "noasan",  # TODO(b/290508632)
    ],
    deps = [
        ":combinations",
        ":device_util",
        ":distribute_lib",
        ":distribute_utils",
        ":input_lib",
        ":input_ops",
        ":input_util",
        ":multi_worker_util",
        ":reduce_util",
        ":strategy_combinations",
        ":test_util",
        "//tensorflow/python:tf2",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
    name = "input_lib_type_spec_test",
    srcs = ["input_lib_type_spec_test.py"],
    main = "input_lib_type_spec_test.py",
    shard_count = 10,
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":distribute_lib",
        ":strategy_combinations",
        ":test_util",
        ":tpu_strategy",
        ":values",
        "//tensorflow/python:tf2",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "cross_device_utils_test",
    srcs = ["cross_device_utils_test.py"],
    deps = [
        ":combinations",
        ":cross_device_utils",
        ":device_util",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "cross_device_ops_test",
    srcs = ["cross_device_ops_test.py"],
    shard_count = 4,
    tags = [
        # "multi_and_single_gpu", # TODO(b/287692888): re-enable once the 2gpu test passes.
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/266520226)
    ],
    deps = [
        ":collective_util",
        ":combinations",
        ":cross_device_ops",
        ":cross_device_utils",
        ":device_util",
        ":multi_process_runner",
        ":multi_worker_test_base",
        ":reduce_util",
        ":test_util",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:collective_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
        "@pypi_dill//:pkg",  # build_cleaner: keep
    ],
)

cuda_py_strict_test(
    name = "one_device_strategy_test",
    srcs = ["one_device_strategy_test.py"],
    grpc_enabled = True,
    deps = [
        ":combinations",
        ":distribute_lib",
        ":strategy_combinations",
        ":strategy_test_lib",
        "//tensorflow/python:tf2",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:device",
    ],
)

py_strict_library(
    name = "sharded_variable",
    srcs = ["sharded_variable.py"],
    deps = [
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "sharded_variable_test",
    size = "small",
    srcs = ["sharded_variable_test.py"],
    tags = [
        "no_mac",
        "no_windows",  # TODO(b/241967911): Re-enable mac and windows.
    ],
    deps = [
        ":combinations",
        ":distribute_lib",
        ":parameter_server_strategy_v2",
        ":sharded_variable",
        ":test_util",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/client:session",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "strategy_test_lib",
    srcs = ["strategy_test_lib.py"],
    deps = [
        ":collective_all_reduce_strategy",
        ":distribute_lib",
        ":distribute_utils",
        ":mirrored_strategy",
        ":reduce_util",
        ":tpu_strategy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:init_ops_v2",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/training:optimizer",
        "//tensorflow/python/training:training_util",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_inspect",
        "//third_party/py/numpy",
    ],
)

distribute_py_strict_test(
    name = "values_test",
    size = "medium",
    srcs = ["values_test.py"],
    main = "values_test.py",
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "no_oss",  # b/178656226
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
        "//tensorflow/python:tf2",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/training:saver",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
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
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/training",
        "//tensorflow/python/types:core",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
    name = "mirrored_values_test",
    size = "medium",
    srcs = ["mirrored_values_test.py"],
    main = "mirrored_values_test.py",
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
    ],
    tpu_tags = [
        "noasan",  # TODO(b/337374867) fails with -fsanitize=null
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        ":tpu_values",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/training:saver",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
    name = "per_replica_test",
    size = "medium",
    srcs = ["per_replica_test.py"],
    main = "per_replica_test.py",
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
    ],
    tpu_tags = [
        "noasan",  # TODO(b/337374867) fails with -fsanitize=null
    ],
    deps = [
        ":combinations",
        ":test_util",
        ":values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
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
        "//tensorflow/python/eager:test",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "@absl_py//absl/testing:parameterized",
        "@pypi_wrapt//:pkg",
    ],
)

distribute_py_strict_test(
    name = "vars_test",
    size = "medium",
    srcs = ["vars_test.py"],
    main = "vars_test.py",
    shard_count = 5,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "no_oss",  # TODO(b/292104274): Sometimes times out
    ],
    deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":distribute_lib",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        ":values",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/training",
        "//tensorflow/python/util:variable_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
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
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
    name = "moving_averages_test",
    srcs = ["moving_averages_test.py"],
    main = "moving_averages_test.py",
    shard_count = 12,
    tags = [
        "multi_gpu",
        "no_windows",  # TODO(b/184424727): Re-enable this.
        "notpu",  # TODO(b/210145904)
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/training:moving_averages",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
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
        "//tensorflow/python:tf2",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
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
        ":device_util",
        ":distribute_lib",
        ":reduce_util",
        ":strategy_combinations",
        ":test_util",
        "//tensorflow/python:tf2",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/tpu:tpu_py",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "single_loss_example",
    srcs = ["single_loss_example.py"],
    deps = [
        ":step_fn",
        ":strategy_test_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/layers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "step_fn",
    srcs = ["step_fn.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/training:optimizer",
    ],
)

cuda_py_strict_test(
    name = "warm_starting_util_test",
    size = "medium",
    srcs = ["warm_starting_util_test.py"],
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/training:warm_starting_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "remote_mirrored_strategy_eager_test",
    srcs = ["remote_mirrored_strategy_eager_test.py"],
    tags = [
        "no_windows",  # TODO(b/197981388): Re-enable this.
    ],
    deps = [
        ":combinations",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":strategy_test_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
)

cuda_py_strict_test(
    name = "mirrored_strategy_test",
    srcs = ["mirrored_strategy_test.py"],
    shard_count = 5,
    tags = [
        "multi_and_single_gpu",
        "no_windows_gpu",  # TODO(b/130551176)
    ],
    deps = [
        ":collective_util",
        ":combinations",
        ":cross_device_ops",
        ":device_util",
        ":distribute_lib",
        ":distribute_utils",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":reduce_util",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:traceback_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "mirrored_variable_test",
    srcs = ["mirrored_variable_test.py"],
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
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:rnn_cell_impl",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
    ],
)

distribute_py_strict_test(
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
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:metrics",
        "//tensorflow/python/ops:variables",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
    name = "zero_batch_test",
    srcs = ["zero_batch_test.py"],
    disable_mlir_bridge = False,
    main = "zero_batch_test.py",
    tags = [
        "no_oss",  # Keras is not available in OSS test
    ],
    deps = [
        ":combinations",
        ":strategy_combinations",
        ":test_util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:gradient_descent",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "collective_all_reduce_strategy_test",
    srcs = ["collective_all_reduce_strategy_test.py"],
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/283098184)
        "notsan",  # TODO(b/220133218)
    ],
    # b/155301154 broken with XLA:GPU
    xla_enable_strict_auto_jit = True,
    deps = [
        ":collective_all_reduce_strategy",
        ":collective_util",
        ":combinations",
        ":distribute_lib",
        ":distribute_utils",
        ":multi_worker_test_base",
        ":multi_worker_util",
        ":reduce_util",
        ":strategy_combinations",
        ":strategy_test_lib",
        ":test_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:init_ops_v2",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "collective_all_reduce_strategy_test_tpu",
    srcs = ["collective_all_reduce_strategy_test.py"],
    main = "collective_all_reduce_strategy_test.py",
    deps = [
        ":collective_all_reduce_strategy",
        ":collective_util",
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
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:init_ops_v2",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "parameter_server_strategy_test",
    srcs = ["parameter_server_strategy_test.py"],
    tags = [
        "multi_and_single_gpu",
        "no_windows",  # TODO(b/284496291)
        "notsan",  # TODO(b/220133218)
    ],
    # b/141096229: Non-atomic AssignAdd
    xla_enable_strict_auto_jit = False,
    deps = [
        ":central_storage_strategy",
        ":combinations",
        ":device_util",
        ":distribute_lib",
        ":distribute_utils",
        ":multi_worker_test_base",
        ":multi_worker_util",
        ":parameter_server_strategy",
        ":ps_values",
        ":reduce_util",
        ":strategy_test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/distribute/v1:input_lib",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:gradients",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:training_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "multi_process_runner",
    srcs = ["multi_process_runner.py"],
    tags = [
        "no_oss",  # b/241013307 disable flaky under docker
    ],
    visibility = [
        "//smartass/brain/ops:__subpackages__",
        "//tensorflow:internal",
    ],
    deps = [
        ":multi_process_lib",
        ":multi_worker_util",
        "//tensorflow/python:tf2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
        "@pypi_dill//:pkg",  # build_cleaner: keep
        "@pypi_tblib//:pkg",  # build_cleaner: keep
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "multi_process_lib",
    srcs = ["multi_process_lib.py"],
    deps = [
        "//tensorflow/python/eager:test",
        "@absl_py//absl:app",
        "@absl_py//absl/logging",
    ],
)

py_strict_test(
    name = "packed_distributed_variable_test",
    srcs = ["packed_distributed_variable_test.py"],
    tags = [
        "nomac",  #TODO(b/145922293): It would cause a Python segfault on macos
    ],
    deps = [
        ":device_util",
        ":packed_distributed_variable",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/trackable:autotrackable",
        "@absl_py//absl/logging",
    ],
)

cuda_py_strict_test(
    name = "multi_process_runner_test",
    srcs = ["multi_process_runner_test.py"],
    shard_count = 35,
    tags = [
        "multi_gpu",
        "no_pip",  # TODO(b/266520226)
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

py_strict_test(
    name = "multi_process_runner_no_init_test",
    srcs = ["multi_process_runner_no_init_test.py"],
    deps = [
        ":multi_process_runner",
        ":multi_worker_test_base",
        "//tensorflow/python/eager:test",
    ],
)

distribute_py_strict_test(
    name = "strategy_common_test",
    srcs = ["strategy_common_test.py"],
    disable_mlir_bridge = False,
    shard_count = 2,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/283033375)
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
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
    name = "strategy_gather_test",
    srcs = ["strategy_gather_test.py"],
    disable_mlir_bridge = False,
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "no_oss",  # TODO(b/283033375)
        "notsan",  # TODO(b/160006974)
        "nozapfhahn",  # TODO(b/317228015)
    ],
    xla_enable_strict_auto_jit = True,
    deps = [
        ":central_storage_strategy",
        ":collective_all_reduce_strategy",
        ":combinations",
        ":distribute_lib",
        ":mirrored_strategy",
        ":strategy_combinations",
        ":test_util",
        ":tpu_strategy",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/saved_model:save_options",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        ":collective_all_reduce_strategy",
        ":multi_process_runner",
        ":multi_worker_test_base",
        ":tpu_strategy",
        ":values",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/util:nest",
        "@absl_py//absl:app",
    ],
)

distribute_py_strict_test(
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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "parameter_server_strategy_v2",
    srcs = ["parameter_server_strategy_v2.py"],
    visibility = [
        "//smartass/brain/ops:__subpackages__",
        "//tensorflow:internal",
    ],
    deps = [
        ":cross_device_ops",
        ":device_util",
        ":distribute_lib",
        ":input_lib",
        ":input_util",
        ":load_context",
        ":mirrored_run",
        ":multi_worker_util",
        ":parameter_server_strategy",
        ":ps_values",
        ":sharded_variable",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/distribute/coordinator:cluster_coordinator",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
    ],
)

distribute_py_strict_test(
    name = "parameter_server_strategy_v2_test",
    srcs = ["parameter_server_strategy_v2_test.py"],
    tags = [
        "multi_and_single_gpu",
        "nomac",  # TODO(b/201788023): Attempt MultiProcessCluster to fix this.
        "notpu",
    ],
    deps = [
        ":distribute_lib",
        ":multi_process_runner",
        ":multi_worker_test_base",
        ":parameter_server_strategy_v2",
        ":ps_values",
        ":sharded_variable",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:init_ops_v2",
        "//tensorflow/python/ops:linalg_ops_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_strict_test(
    name = "distributed_table_test",
    srcs = ["distributed_table_test.py"],
    shard_count = 12,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/249822228)
        "noasan",  # TODO(b/237407459)
        "nomsan",  # TODO(b/290745680)
        "notpu",
        "notsan",  # Tsan failure doesn't seem to be caused by TF.
    ],
    deps = [
        ":combinations",
        ":device_util",
        ":multi_process_runner",
        ":multi_worker_test_base",
        ":parameter_server_strategy_v2",
        ":ps_values",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/coordinator:cluster_coordinator",
        "//tensorflow/python/distribute/coordinator:coordinator_context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_strategy_model_parallelism_test",
    srcs = ["tpu_strategy_model_parallelism_test.py"],
    args = [
        "--xla_tpu_reserve_recv_ready_sync_flags=true",
    ],
    disable_experimental = True,  # b/202779350
    disable_mlir_bridge = False,
    disable_v3_4chips = False,
    tags = ["no_oss"],
    deps = [
        ":distribute_lib",
        ":packed_distributed_variable",
        ":strategy_test_lib",
        ":tpu_replicated_variable",
        ":tpu_strategy",
        ":tpu_values",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/compiler/xla/experimental:xla_sharding",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:summary_test_util",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/tpu:device_assignment",
        "//tensorflow/python/tpu:tpu_replication",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "input_util",
    srcs = ["input_util.py"],
    deps = [
        ":input_lib",
        "//tensorflow/python:tf2",
        "//tensorflow/python/distribute/v1:input_lib",
    ],
)

pytype_strict_library(
    name = "load_context",
    srcs = ["load_context.py"],
    deps = ["//tensorflow/python/util:tf_export"],
)

py_strict_library(
    name = "merge_call_interim",
    srcs = [
        "merge_call_interim.py",
    ],
    deps = [
        ":distribute_lib",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "template_mirrored_strategy_test",
    size = "small",
    srcs = ["template_mirrored_strategy_test.py"],
    deps = [
        ":distribute_lib",
        ":mirrored_strategy",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:template",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)
