load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow/core/platform/default:distribute.bzl", "distribute_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

exports_files(
    # Used in a pybind extension whose rule must be in tensorflow/python
    ["quantize_training_wrapper.cc"],
    visibility = ["//tensorflow/python:__pkg__"],
)

exports_files(
    # Used in a rule which visibility limits to tensorflow/python
    ["learning_rate_decay.py"],
    visibility = ["//tensorflow/python:__pkg__"],
)

# Files which have their own BUILD rules, but which for compatibility with
# strict dep checking need to be direct dependencies of training_lib. Do not add
# any new files to this list.
filegroup(
    name = "deprecated_inclusions_in_training_lib",
    srcs = [
        "adadelta.py",
        "adagrad.py",
        "adagrad_da.py",
        "adam.py",
        "basic_loops.py",
        "checkpoint_ops.py",
        "checkpoint_utils.py",
        "coordinator.py",
        "device_setter.py",
        "evaluation.py",
        "ftrl.py",
        "gradient_descent.py",
        "input.py",
        "learning_rate_decay.py",
        "momentum.py",
        "monitored_session.py",
        "moving_averages.py",
        "optimizer.py",
        "proximal_adagrad.py",
        "proximal_gradient_descent.py",
        "py_checkpoint_reader.py",
        "quantize_training.py",
        "queue_runner.py",
        "queue_runner_impl.py",
        "rmsprop.py",
        "server_lib.py",
        "session_manager.py",
        "slot_creator.py",
        "summary_io.py",
        "supervisor.py",
        "sync_replicas_optimizer.py",
        "training.py",
        "training_ops.py",
        "warm_starting_util.py",
    ],
    visibility = ["//visibility:private"],
)

py_library(
    name = "training_lib",
    srcs = [
        "__init__.py",
        "training.py",
        ":deprecated_inclusions_in_training_lib",
    ],
    srcs_version = "PY3",
    deps = [
        ":adadelta",
        ":adagrad",
        ":adagrad_da",
        ":adam",
        ":basic_loops",
        ":basic_session_run_hooks",
        ":checkpoint_management",
        ":checkpoint_utils",
        ":coordinator",
        ":device_setter",
        ":ftrl",
        ":gradient_descent",
        ":input",
        ":momentum",
        ":monitored_session",
        ":moving_averages",
        ":optimizer",
        ":proximal_adagrad",
        ":proximal_gradient_descent",
        ":py_checkpoint_reader",
        ":quantize_training",
        ":queue_runner",
        ":rmsprop",
        ":saver",
        ":server_lib",
        ":session_manager",
        ":session_run_hook",
        ":summary_io",
        ":supervisor",
        ":sync_replicas_optimizer",
        ":training_util",
        ":warm_starting_util",
        "//tensorflow/python:learning_rate_decay",
        "//tensorflow/python:sdca_ops",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/trackable:base_delegate",
        "//tensorflow/python/training/experimental:loss_scale_optimizer",
        "//tensorflow/python/training/experimental:mixed_precision",
        "//tensorflow/python/training/tracking:base_delegate",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "training",
    srcs_version = "PY3",
    deps = [
        ":training_lib",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:python_state",
        "//tensorflow/python/training/tracking:base",
        "//tensorflow/python/training/tracking:python_state",
        "//tensorflow/python/training/tracking:util",
    ],
)

py_library(
    name = "adadelta",
    srcs = ["adadelta.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":training_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "adagrad_da",
    srcs = ["adagrad_da.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":training_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "adagrad",
    srcs = ["adagrad.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":training_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "adam",
    srcs = ["adam.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":training_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "basic_loops",
    srcs = ["basic_loops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "checkpoint_ops",
    srcs = ["checkpoint_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:checkpoint_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":py_checkpoint_reader",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_library(
    name = "coordinator",
    srcs = ["coordinator.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_library(
    name = "device_setter",
    srcs = ["device_setter.py"],
    srcs_version = "PY3",
    deps = [
        ":server_lib",
        "//tensorflow/python:device",
        "//tensorflow/python:platform",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_library(
    name = "distribution_strategy_context",
    srcs = ["distribution_strategy_context.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python/distribute:distribute_lib"],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    srcs_version = "PY3",
    deps = [
        ":basic_session_run_hooks",
        ":monitored_session",
        ":session_run_hook",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "ftrl",
    srcs = ["ftrl.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":training_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "gradient_descent",
    srcs = ["gradient_descent.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":training_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "input",
    srcs = ["input.py"],
    srcs_version = "PY3",
    deps = [
        ":queue_runner",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:layers_util",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/summary",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_library(
    name = "momentum",
    srcs = ["momentum.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":training_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "moving_averages",
    srcs = ["moving_averages.py"],
    srcs_version = "PY3",
    deps = [
        ":slot_creator",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    srcs_version = "PY3",
    deps = [
        ":slot_creator",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:distribute_utils",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_library(
    name = "proximal_adagrad",
    srcs = ["proximal_adagrad.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":training_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "proximal_gradient_descent",
    srcs = ["proximal_gradient_descent.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":training_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "quantize_training",
    srcs = ["quantize_training.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:_pywrap_quantize_training",
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "queue_runner_impl",
    srcs = ["queue_runner_impl.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "queue_runner",
    srcs = ["queue_runner.py"],
    srcs_version = "PY3",
    deps = [":queue_runner_impl"],
)

py_library(
    name = "rmsprop",
    srcs = ["rmsprop.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":training_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "session_manager",
    srcs = ["session_manager.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python/checkpoint:checkpoint_lib",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "slot_creator",
    srcs = ["slot_creator.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/xla/experimental/xla_sharding",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
    ],
)

py_library(
    name = "summary_io",
    srcs = ["summary_io.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/summary",
    ],
)

py_library(
    name = "sync_replicas_optimizer",
    srcs = ["sync_replicas_optimizer.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":queue_runner",
        ":session_manager",
        ":session_run_hook",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "training_ops",
    srcs = [
        "gen_training_ops.py",
        "training_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:training_ops_gen",
    ],
)

py_library(
    name = "warm_starting_util",
    srcs = ["warm_starting_util.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_ops",
        ":checkpoint_utils",
        ":saver",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_library(
    name = "distribute",
    srcs = [
        "distribute.py",
        "distribution_strategy_context.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/distribute:distribute_lib",
    ],
)

tf_py_test(
    name = "server_lib_test",
    size = "small",
    srcs = ["server_lib_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    tags = [
        "noasan",  # TODO(b/161236904): flaky timeout in trying to start gRPC server
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "server_lib_multiple_containers_test",
    size = "small",
    srcs = ["server_lib_multiple_containers_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "server_lib_same_variables_clear_container_test",
    size = "small",
    srcs = ["server_lib_same_variables_clear_container_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "server_lib_same_variables_clear_test",
    size = "small",
    srcs = ["server_lib_same_variables_clear_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "server_lib_same_variables_no_clear_test",
    size = "small",
    srcs = ["server_lib_same_variables_no_clear_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "server_lib_sparse_job_test",
    size = "small",
    srcs = ["server_lib_sparse_job_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "localhost_cluster_performance_test",
    size = "medium",
    srcs = [
        "localhost_cluster_performance_test.py",
    ],
    grpc_enabled = True,
    python_version = "PY3",
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "oss_serial",
    ],
    deps = [
        ":device_setter",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:session",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sync_replicas_optimizer_test",
    size = "medium",
    srcs = [
        "sync_replicas_optimizer_test.py",
    ],
    grpc_enabled = True,
    python_version = "PY3",
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "notsan",  # data race due to b/62910646
        "oss_serial",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "evaluation_test",
    size = "small",
    srcs = ["evaluation_test.py"],
    python_version = "PY3",
    shard_count = 3,
    tags = [
        "manual",
        "notap",  # Disabling until b/33000128 and b/33040312 are fixed.
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/summary",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "py_checkpoint_reader",
    srcs = ["py_checkpoint_reader.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:util",
        "//tensorflow/python/util:_pywrap_checkpoint_reader",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_proto_library(
    name = "checkpoint_state",
    srcs = ["checkpoint_state.proto"],
    cc_api_version = 2,
)

py_library(
    name = "checkpoint_management",
    srcs = ["checkpoint_management.py"],
    srcs_version = "PY3",
    deps = [
        ":training_util",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "saver",
    srcs = ["saver.py"],
    srcs_version = "PY3",
    deps = [
        ":py_checkpoint_reader",
        ":training_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/saved_model:pywrap_saved_model",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "saver_test_utils",
    srcs = ["saver_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":saver",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops_gen",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "saver_test",
    size = "medium",
    srcs = [
        "saver_test.py",
    ],
    python_version = "PY3",
    tags = ["multi_gpu"],
    deps = [
        ":adam",
        ":gradient_descent",
        ":py_checkpoint_reader",
        ":queue_runner_impl",
        ":saver",
        ":saver_test_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:function",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/summary",
        "//tensorflow/python/trackable:base",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "saver_large_variable_test",
    size = "medium",
    srcs = ["saver_large_variable_test.py"],
    python_version = "PY3",
    tags = [
        "manual",
        "noasan",  # http://b/30379628
        "notsan",  # http://b/30379628
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "saver_large_partitioned_variable_test",
    size = "medium",
    srcs = ["saver_large_partitioned_variable_test.py"],
    python_version = "PY3",
    tags = [
        "noasan",  # http://b/30782289
        "notsan",  # http://b/30782289
    ],
    deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "basic_session_run_hooks",
    srcs = ["basic_session_run_hooks.py"],
    srcs_version = "PY3",
    deps = [
        ":session_run_hook",
        ":summary_io",
        ":training_util",
        "//tensorflow/python:client",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "session_run_hook",
    srcs = ["session_run_hook.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python/util:tf_export"],
)

py_library(
    name = "supervisor",
    srcs = ["supervisor.py"],
    srcs_version = "PY3",
    deps = [
        ":coordinator",
        ":saver",
        ":session_manager",
        ":training_util",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/summary",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "supervisor_test",
    size = "small",
    srcs = ["supervisor_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    tags = ["no_windows"],
    deps = [
        ":saver",
        ":supervisor",
        ":training",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/summary",
    ],
)

py_library(
    name = "server_lib",
    srcs = ["server_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "training_util",
    srcs = ["training_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "training_util_test",
    size = "small",
    srcs = ["training_util_test.py"],
    python_version = "PY3",
    deps = [
        ":training_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "adam_test",
    size = "medium",
    srcs = ["adam_test.py"],
    python_version = "PY3",
    tags = ["no_rocm"],
    deps = [
        ":adam",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "moving_averages_test",
    size = "small",
    srcs = [
        "moving_averages_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_windows",  # b/139083295: bfloat16 tests fail on Windows
        "notsan",
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":moving_averages",
        ":saver",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
    ],
)

TRAINING_TEST_DEPS = [
    "//third_party/py/numpy",
    "@six_archive//:six",
    "//tensorflow/core:protos_all_py",
    "//tensorflow/python:array_ops",
    "//tensorflow/python:client",
    "//tensorflow/python:client_testlib",
    "//tensorflow/python:control_flow_ops",
    "//tensorflow/python:data_flow_ops",
    "//tensorflow/python:data_flow_ops_gen",
    "//tensorflow/python:embedding_ops",
    "//tensorflow/python:errors",
    "//tensorflow/python:framework",
    "//tensorflow/python:framework_for_generated_wrappers",
    "//tensorflow/python:framework_test_lib",
    "//tensorflow/python:gradients",
    "//tensorflow/python:lookup_ops",
    "//tensorflow/python:math_ops",
    "//tensorflow/python:nn_grad",
    "//tensorflow/python:nn_ops",
    "//tensorflow/python:partitioned_variables",
    "//tensorflow/python:platform",
    "//tensorflow/python:platform_test",
    "//tensorflow/python:pywrap_tensorflow",
    "//tensorflow/python:random_ops",
    "//tensorflow/python:resource_variable_ops",
    "//tensorflow/python:resources",
    "//tensorflow/python:sparse_ops",
    "//tensorflow/python:state_ops",
    "//tensorflow/python:state_ops_gen",
    "//tensorflow/python:util",
    "//tensorflow/python:variable_scope",
    "//tensorflow/python:variables",
    "//tensorflow/python/distribute:cross_device_ops",
    "//tensorflow/python/distribute:distribute_utils",
    "//tensorflow/python/distribute:mirrored_strategy",
    "//tensorflow/python/summary",
]

cuda_py_test(
    name = "adadelta_test",
    size = "medium",
    srcs = ["adadelta_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "adagrad_da_test",
    size = "medium",
    srcs = ["adagrad_da_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "adagrad_test",
    size = "medium",
    srcs = ["adagrad_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "basic_loops_test",
    size = "medium",
    srcs = ["basic_loops_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "coordinator_test",
    size = "medium",
    srcs = ["coordinator_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "device_setter_test",
    size = "medium",
    srcs = ["device_setter_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "ftrl_test",
    size = "medium",
    srcs = ["ftrl_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "gradient_descent_test",
    size = "medium",
    srcs = ["gradient_descent_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "momentum_test",
    size = "medium",
    srcs = ["momentum_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "optimizer_test",
    size = "medium",
    srcs = ["optimizer_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "proximal_adagrad_test",
    size = "medium",
    srcs = ["proximal_adagrad_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "proximal_gradient_descent_test",
    size = "medium",
    srcs = ["proximal_gradient_descent_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "quantize_training_test",
    size = "medium",
    srcs = ["quantize_training_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "queue_runner_test",
    size = "medium",
    srcs = ["queue_runner_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "rmsprop_test",
    size = "medium",
    srcs = ["rmsprop_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "slot_creator_test",
    size = "medium",
    srcs = ["slot_creator_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_test(
    name = "training_ops_test",
    size = "medium",
    srcs = ["training_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":training",
    ] + TRAINING_TEST_DEPS,
)

test_suite(
    name = "training_tests",
    tests = [
        "adadelta_test",
        "adagrad_da_test",
        "adagrad_test",
        "basic_loops_test",
        "coordinator_test",
        "device_setter_test",
        "ftrl_test",
        "gradient_descent_test",
        "momentum_test",
        "optimizer_test",
        "proximal_adagrad_test",
        "proximal_gradient_descent_test",
        "quantize_training_test",
        "queue_runner_test",
        "rmsprop_test",
        "slot_creator_test",
        "training_ops_test",
    ],
)

distribute_py_test(
    name = "training_ops_mlir_test",
    srcs = [
        "training_ops_test.py",
    ],
    disable_mlir_bridge = False,
    main = "training_ops_test.py",
    deps = [
        ":training",
    ],
)

cuda_py_test(
    name = "session_manager_test",
    size = "medium",  # TODO(irving): Can this be made small?
    srcs = ["session_manager_test.py"],
    grpc_enabled = True,
    main = "session_manager_test.py",
    python_version = "PY3",
    deps = [
        ":saver",
        ":server_lib",
        ":session_manager",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint:checkpoint_management",
    ],
)

tf_py_test(
    name = "basic_session_run_hooks_test",
    size = "medium",
    srcs = ["basic_session_run_hooks_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",  # Relies on contrib
        "no_windows",
        "notsan",  # intermittent races on a few percent of runs
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/summary",
        "//tensorflow/python/summary/writer",
        "//tensorflow/python/summary/writer:fake_summary_writer",
    ],
)

tf_py_test(
    name = "checkpoint_utils_test",
    size = "small",
    srcs = ["checkpoint_utils_test.py"],
    python_version = "PY3",
    tags = [
        "manual",
        "no_cuda_on_cpu_tap",
        "no_oss",
        "no_windows",
        "notap",
    ],
    deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "checkpoint_ops_test",
    size = "small",
    srcs = ["checkpoint_ops_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:checkpoint_ops_gen",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "warm_starting_util_test",
    size = "medium",
    srcs = ["warm_starting_util_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "monitored_session",
    srcs = ["monitored_session.py"],
    srcs_version = "PY3",
    deps = [
        ":basic_session_run_hooks",
        ":coordinator",
        ":queue_runner",
        ":saver",
        ":session_manager",
        ":session_run_hook",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resources",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_coordinator_context",
        "//tensorflow/python/summary",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "monitored_session_test",
    size = "medium",
    srcs = ["monitored_session_test.py"],
    tags = [
        "no_pip",
        "notsan",  # b/67945581
    ],
    deps = [
        ":monitored_session",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/summary",
    ],
)

tf_py_test(
    name = "input_test",
    size = "medium",
    srcs = ["input_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)
