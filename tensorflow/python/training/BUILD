load("//tensorflow:strict.default.bzl", "py_strict_library")

# Placeholder: load py_proto_library
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/core/platform:distribute.bzl", "distribute_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

exports_files(
    # Used in a pybind extension whose rule must be in tensorflow/python
    ["quantize_training_wrapper.cc"],
    visibility = ["//tensorflow/python:__pkg__"],
)

py_strict_library(
    name = "training_lib",
    srcs = ["training.py"],
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":adadelta",
        ":adagrad",
        ":adagrad_da",
        ":adam",
        ":basic_loops",
        ":basic_session_run_hooks",
        ":checkpoint_utils",
        ":coordinator",
        ":device_setter",
        ":ftrl",
        ":gradient_descent",
        ":input",
        ":learning_rate_decay",
        ":momentum",
        ":monitored_session",
        ":moving_averages",
        ":optimizer",
        ":proximal_adagrad",
        ":proximal_gradient_descent",
        ":py_checkpoint_reader",
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
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/checkpoint:checkpoint_view",
        "//tensorflow/python/ops:sdca_ops",
        "//tensorflow/python/trackable:python_state",
        "//tensorflow/python/training/experimental:loss_scale_optimizer",
        "//tensorflow/python/training/experimental:mixed_precision",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "training",
    visibility = [
        "//tensorflow:internal",
        "//tensorflow_minigo:__subpackages__",
        "//tensorflow_models:__subpackages__",
        "//third_party/cloud_tpu/convergence_tools:__subpackages__",
        "//third_party/mlperf:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":training_lib",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:python_state",
    ],
)

py_strict_library(
    name = "adadelta",
    srcs = ["adadelta.py"],
    deps = [
        ":optimizer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "adagrad_da",
    srcs = ["adagrad_da.py"],
    deps = [
        ":optimizer",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "adagrad",
    srcs = ["adagrad.py"],
    deps = [
        ":optimizer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "adam",
    srcs = ["adam.py"],
    deps = [
        ":optimizer",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "basic_loops",
    srcs = ["basic_loops.py"],
    deps = [
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "checkpoint_ops",
    srcs = ["checkpoint_ops.py"],
    deps = [
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:checkpoint_ops_gen",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    deps = [
        ":py_checkpoint_reader",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "coordinator",
    srcs = ["coordinator.py"],
    deps = [
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "device_setter",
    srcs = ["device_setter.py"],
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim/ops:__pkg__",
    ],
    deps = [
        ":server_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim/training:__pkg__",
    ],
    deps = [
        ":basic_session_run_hooks",
        ":monitored_session",
        ":session_run_hook",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "ftrl",
    srcs = ["ftrl.py"],
    deps = [
        ":optimizer",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "gradient_descent",
    srcs = ["gradient_descent.py"],
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim/layers:__pkg__",
        "//third_party/py/tf_slim/training:__pkg__",
    ],
    deps = [
        ":optimizer",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "input",
    srcs = ["input.py"],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/contrib/training:__pkg__",
    ],
    deps = [
        ":queue_runner",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/layers:layers_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "momentum",
    srcs = ["momentum.py"],
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim/losses:__pkg__",
    ],
    deps = [
        ":optimizer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "moving_averages",
    srcs = ["moving_averages.py"],
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim/layers:__pkg__",
    ],
    deps = [
        ":slot_creator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_strict_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim/layers:__pkg__",
        "//third_party/py/tf_slim/training:__pkg__",
    ],
    deps = [
        ":slot_creator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:distribute_utils",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:gradients",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "proximal_adagrad",
    srcs = ["proximal_adagrad.py"],
    deps = [
        ":optimizer",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "proximal_gradient_descent",
    srcs = ["proximal_gradient_descent.py"],
    deps = [
        ":optimizer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "quantize_training",
    srcs = ["quantize_training.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:_pywrap_quantize_training",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "queue_runner_impl",
    srcs = ["queue_runner_impl.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "queue_runner",
    srcs = ["queue_runner.py"],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/contrib/training:__pkg__",
    ],
    deps = [":queue_runner_impl"],
)

py_strict_library(
    name = "rmsprop",
    srcs = ["rmsprop.py"],
    deps = [
        ":optimizer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "session_manager",
    srcs = ["session_manager.py"],
    deps = [
        ":saver",
        "//tensorflow/python/checkpoint:checkpoint_lib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "slot_creator",
    srcs = ["slot_creator.py"],
    deps = [
        "//tensorflow/python/compiler/xla/experimental:xla_sharding",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:ref_variable",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
    ],
)

py_strict_library(
    name = "summary_io",
    srcs = ["summary_io.py"],
    deps = [
        "//tensorflow/python/summary:summary_iterator",
        "//tensorflow/python/summary/writer",
        "//tensorflow/python/summary/writer:writer_cache",
        "//tensorflow/python/util:deprecation",
    ],
)

py_strict_library(
    name = "sync_replicas_optimizer",
    srcs = ["sync_replicas_optimizer.py"],
    deps = [
        ":optimizer",
        ":queue_runner",
        ":session_manager",
        ":session_run_hook",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "warm_starting_util",
    srcs = ["warm_starting_util.py"],
    deps = [
        ":checkpoint_ops",
        ":checkpoint_utils",
        ":saver",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "server_lib_test",
    size = "small",
    srcs = ["server_lib_test.py"],
    grpc_enabled = True,
    deps = [
        ":input",
        ":queue_runner_impl",
        ":server_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "server_lib_multiple_containers_test",
    size = "small",
    srcs = ["server_lib_multiple_containers_test.py"],
    grpc_enabled = True,
    deps = [
        ":server_lib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "server_lib_same_variables_clear_container_test",
    size = "small",
    srcs = ["server_lib_same_variables_clear_container_test.py"],
    grpc_enabled = True,
    deps = [
        ":server_lib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "server_lib_same_variables_clear_test",
    size = "small",
    srcs = ["server_lib_same_variables_clear_test.py"],
    grpc_enabled = True,
    deps = [
        ":server_lib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "server_lib_same_variables_no_clear_test",
    size = "small",
    srcs = ["server_lib_same_variables_no_clear_test.py"],
    grpc_enabled = True,
    deps = [
        ":server_lib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "server_lib_sparse_job_test",
    size = "small",
    srcs = ["server_lib_sparse_job_test.py"],
    grpc_enabled = True,
    deps = [
        ":server_lib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "localhost_cluster_performance_test",
    size = "medium",
    srcs = [
        "localhost_cluster_performance_test.py",
    ],
    grpc_enabled = True,
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "oss_serial",
    ],
    deps = [
        ":device_setter",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "sync_replicas_optimizer_test",
    size = "medium",
    srcs = [
        "sync_replicas_optimizer_test.py",
    ],
    grpc_enabled = True,
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "oss_serial",
    ],
    deps = [
        ":adam",
        ":gradient_descent",
        ":training_lib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "evaluation_test",
    size = "small",
    srcs = ["evaluation_test.py"],
    shard_count = 3,
    tags = [
        "manual",
        "notap",  # Disabling until b/33000128 and b/33040312 are fixed.
    ],
    deps = [
        ":basic_session_run_hooks",
        ":evaluation",
        ":gradient_descent",
        ":monitored_session",
        ":saver",
        ":training_lib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:metrics",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "py_checkpoint_reader",
    srcs = ["py_checkpoint_reader.py"],
    deps = [
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/util:_pywrap_checkpoint_reader",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_proto_library(
    name = "checkpoint_state",
    srcs = ["checkpoint_state.proto"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "checkpoint_state_py_pb2",
#     testonly = 0,
#     deps = [":checkpoint_state"],
# )
# copybara:uncomment_end

py_strict_library(
    name = "checkpoint_management",
    srcs = ["checkpoint_management.py"],
    deps = [
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/util:deprecation",
    ],
)

py_strict_library(
    name = "saver",
    srcs = ["saver.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = [
    # "//third_party/py/tf_slim:__subpackages__",
    # "//tensorflow:internal",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":py_checkpoint_reader",
        ":training_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/ops:io_ops_gen",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:pywrap_saved_model",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "learning_rate_decay",
    srcs = ["learning_rate_decay.py"],
    deps = ["//tensorflow/python/keras/optimizer_v2:legacy_learning_rate_decay"],
)

py_strict_library(
    name = "saver_test_utils",
    srcs = ["saver_test_utils.py"],
    deps = [
        ":saver",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:lookup_ops_gen",
    ],
)

cuda_py_strict_test(
    name = "saver_test",
    size = "medium",
    srcs = [
        "saver_test.py",
    ],
    tags = ["multi_gpu"],
    deps = [
        ":adam",
        ":gradient_descent",
        ":py_checkpoint_reader",
        ":queue_runner_impl",
        ":saver",
        ":saver_test_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:graph_io",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/saved_model:pywrap_saved_model",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util:compat",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "saver_large_variable_test",
    size = "medium",
    srcs = ["saver_large_variable_test.py"],
    tags = ["manual"],
    deps = [
        ":saver",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "saver_large_partitioned_variable_test",
    size = "medium",
    srcs = ["saver_large_partitioned_variable_test.py"],
    deps = [
        ":saver",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "basic_session_run_hooks",
    srcs = ["basic_session_run_hooks.py"],
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":session_run_hook",
        ":summary_io",
        ":training_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:timeline",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "session_run_hook",
    srcs = ["session_run_hook.py"],
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = ["//tensorflow/python/util:tf_export"],
)

py_strict_library(
    name = "supervisor",
    srcs = ["supervisor.py"],
    deps = [
        ":coordinator",
        ":saver",
        ":session_manager",
        ":training_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "supervisor_test",
    size = "small",
    srcs = ["supervisor_test.py"],
    grpc_enabled = True,
    tags = ["no_windows"],
    deps = [
        ":input",
        ":saver",
        ":server_lib",
        ":session_manager",
        ":supervisor",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/summary:summary_iterator",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/summary/writer",
    ],
)

py_strict_library(
    name = "server_lib",
    srcs = ["server_lib.py"],
    visibility = [
        "//smartass/brain/ops:__subpackages__",
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "training_util",
    srcs = ["training_util.py"],
    visibility = [
        "//tensorflow:internal",
        "//third_party/cloud_tpu/convergence_tools:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:graph_io",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "training_util_test",
    size = "small",
    srcs = ["training_util_test.py"],
    deps = [
        ":monitored_session",
        ":training_util",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "adam_test",
    size = "medium",
    srcs = ["adam_test.py"],
    tags = ["no_rocm"],
    deps = [
        ":adam",
        "//tensorflow/python/client:session",
        "//tensorflow/python/compiler/xla/experimental:xla_sharding",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:ref_variable",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "moving_averages_test",
    size = "small",
    srcs = [
        "moving_averages_test.py",
    ],
    tags = [
        "no_windows",  # b/139083295: bfloat16 tests fail on Windows
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":moving_averages",
        ":saver",
        "//tensorflow/python/compiler/xla/experimental:xla_sharding",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:state_ops_gen",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

TRAINING_TEST_DEPS = [
    "//third_party/py/numpy",
    "//tensorflow/core:protos_all_py",
    "//tensorflow/python/ops:array_ops",
    "//tensorflow/python/client:client",
    "//tensorflow/python/platform:client_testlib",
    "//tensorflow/python/ops:cond",
    "//tensorflow/python/ops:control_flow_ops",
    "//tensorflow/python/ops:data_flow_ops",
    "//tensorflow/python/ops:data_flow_ops_gen",
    "//tensorflow/python/ops:embedding_ops",
    "//tensorflow/python/framework:errors",
    "//tensorflow/python/framework:for_generated_wrappers",
    "//tensorflow/python/framework:test_lib",
    "//tensorflow/python/ops:custom_gradient",
    "//tensorflow/python/ops:gradients",
    "//tensorflow/python/ops:lookup_ops",
    "//tensorflow/python/ops:math_ops",
    "//tensorflow/python/ops:nn_grad",
    "//tensorflow/python/ops:nn_ops",
    "//tensorflow/python/ops:partitioned_variables",
    "//tensorflow/python/platform:test",
    "//tensorflow/python:pywrap_tensorflow",
    "//tensorflow/python/ops:random_ops",
    "//tensorflow/python/ops:resource_variable_ops",
    "//tensorflow/python/ops:resources",
    "//tensorflow/python/ops:sparse_ops",
    "//tensorflow/python/ops:state_ops",
    "//tensorflow/python/ops:state_ops_gen",
    "//tensorflow/python/ops:variable_scope",
    "//tensorflow/python/ops:variables",
    "//tensorflow/python/distribute:cross_device_ops",
    "//tensorflow/python/distribute:distribute_utils",
    "//tensorflow/python/distribute:mirrored_strategy",
]

cuda_py_strict_test(
    name = "adadelta_test",
    size = "medium",
    srcs = ["adadelta_test.py"],
    deps = [
        ":adadelta",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "adagrad_da_test",
    size = "medium",
    srcs = ["adagrad_da_test.py"],
    deps = [
        ":adagrad_da",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "adagrad_test",
    size = "medium",
    srcs = ["adagrad_test.py"],
    deps = [
        ":adagrad",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "basic_loops_test",
    size = "medium",
    srcs = ["basic_loops_test.py"],
    deps = [
        ":basic_loops",
        ":supervisor",
        "//tensorflow/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "coordinator_test",
    size = "medium",
    srcs = ["coordinator_test.py"],
    deps = [
        ":coordinator",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "device_setter_test",
    size = "medium",
    srcs = ["device_setter_test.py"],
    deps = [
        ":device_setter",
        ":server_lib",
        "//tensorflow/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "ftrl_test",
    size = "medium",
    srcs = ["ftrl_test.py"],
    deps = [
        ":adagrad",
        ":ftrl",
        ":gradient_descent",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "gradient_descent_test",
    size = "medium",
    srcs = ["gradient_descent_test.py"],
    deps = [
        ":gradient_descent",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "momentum_test",
    size = "medium",
    srcs = ["momentum_test.py"],
    deps = [
        ":momentum",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "optimizer_test",
    size = "medium",
    srcs = ["optimizer_test.py"],
    deps = [
        ":adam",
        ":gradient_descent",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:clip_ops",
        "//tensorflow/python/ops:gradients_util",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "proximal_adagrad_test",
    size = "medium",
    srcs = ["proximal_adagrad_test.py"],
    deps = [
        ":adagrad",
        ":proximal_adagrad",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "proximal_gradient_descent_test",
    size = "medium",
    srcs = ["proximal_gradient_descent_test.py"],
    deps = [
        ":gradient_descent",
        ":proximal_gradient_descent",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "quantize_training_test",
    size = "medium",
    srcs = ["quantize_training_test.py"],
    deps = [
        ":quantize_training",
        ":saver",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variable_v1",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "queue_runner_test",
    size = "medium",
    srcs = ["queue_runner_test.py"],
    deps = [
        ":coordinator",
        ":monitored_session",
        ":queue_runner_impl",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variable_v1",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "rmsprop_test",
    size = "medium",
    srcs = ["rmsprop_test.py"],
    deps = [
        ":rmsprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "slot_creator_test",
    size = "medium",
    srcs = ["slot_creator_test.py"],
    deps = [
        ":slot_creator",
        "//tensorflow/python/compiler/xla/experimental:xla_sharding",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variable_v1",
        "@local_xla//xla:xla_data_proto_py",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "training_ops_test",
    size = "medium",
    srcs = ["training_ops_test.py"],
    deps = [
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/ops:variable_v1",
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

distribute_py_strict_test(
    name = "training_ops_mlir_test",
    srcs = [
        "training_ops_test.py",
    ],
    disable_mlir_bridge = False,
    main = "training_ops_test.py",
    deps = [
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:training_ops_gen",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:test",
    ],
)

cuda_py_strict_test(
    name = "session_manager_test",
    size = "medium",  # TODO(irving): Can this be made small?
    srcs = ["session_manager_test.py"],
    grpc_enabled = True,
    main = "session_manager_test.py",
    deps = [
        ":saver",
        ":server_lib",
        ":session_manager",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:resource_variables_toggle",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
    ],
)

tf_py_strict_test(
    name = "basic_session_run_hooks_test",
    size = "medium",
    srcs = ["basic_session_run_hooks_test.py"],
    tags = [
        "no_pip",  # Relies on contrib
        "no_windows",
        "notsan",  # intermittent races on a few percent of runs
    ],
    deps = [
        ":basic_session_run_hooks",
        ":checkpoint_utils",
        ":monitored_session",
        ":session_run_hook",
        ":training_util",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/summary/writer:fake_summary_writer",
        "//tensorflow/python/summary/writer:writer_cache",
    ],
)

tf_py_strict_test(
    name = "checkpoint_utils_iterator_test",
    size = "small",
    srcs = ["checkpoint_utils_iterator_test.py"],
    deps = [
        ":checkpoint_utils",
        ":saver",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
    ],
)

tf_py_strict_test(
    name = "checkpoint_utils_test",
    size = "small",
    srcs = ["checkpoint_utils_test.py"],
    tags = [
        "manual",
        "no_cuda_on_cpu_tap",
        "no_oss",
        "no_windows",
        "notap",
    ],
    deps = [
        ":checkpoint_utils",
        ":saver",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
    ],
)

tf_py_strict_test(
    name = "checkpoint_ops_test",
    size = "small",
    srcs = ["checkpoint_ops_test.py"],
    deps = [
        ":checkpoint_ops",
        ":saver",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "warm_starting_util_test",
    size = "medium",
    srcs = ["warm_starting_util_test.py"],
    deps = [
        ":checkpoint_utils",
        ":saver",
        ":warm_starting_util",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "monitored_session",
    srcs = ["monitored_session.py"],
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim/training:__pkg__",
    ],
    deps = [
        ":basic_session_run_hooks",
        ":coordinator",
        ":queue_runner",
        ":saver",
        ":session_manager",
        ":session_run_hook",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:graph_view",
        "//tensorflow/python/distribute:distribute_coordinator_context",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:resources",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/util:function_utils",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "monitored_session_test",
    size = "medium",
    srcs = ["monitored_session_test.py"],
    tags = [
        "no_pip",
        "notsan",  # b/67945581
    ],
    deps = [
        ":basic_session_run_hooks",
        ":coordinator",
        ":monitored_session",
        ":saver",
        ":session_run_hook",
        ":summary_io",
        ":training_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/client:session",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/summary:summary_py",
    ],
)

tf_py_strict_test(
    name = "input_test",
    size = "medium",
    srcs = ["input_test.py"],
    deps = [
        ":coordinator",
        ":input",
        ":queue_runner_impl",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//third_party/py/numpy",
    ],
)
