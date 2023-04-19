# Description:
#   Utilities for reading and writing object-based checkpoints.

load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library")
load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_py_logged_benchmark",
)
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "checkpoint_lib",
    deps = [
        ":checkpoint_core",
        ":checkpoint_management",
        ":checkpoint_options",
        ":functional_saver",
        ":graph_view",
        ":saveable_compat",
        ":util",
    ],
)

py_strict_library(
    name = "async_checkpoint_helper",
    srcs = ["async_checkpoint_helper.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_context",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:executor",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/saved_model:pywrap_saved_model",
        "//tensorflow/python/util:object_identity",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "checkpoint",
    srcs_version = "PY3",
    deps = [
        ":async_checkpoint_helper",
        ":checkpoint_core",
    ],
)

py_strict_library(
    name = "checkpoint_core",
    srcs = [
        "checkpoint.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_context",
        ":checkpoint_management",
        ":checkpoint_options",
        ":functional_saver",
        ":graph_view",
        ":restore",
        ":save_util",
        ":save_util_v1",
        ":util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:lib",
        "//tensorflow/python:saver",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:path_helpers",
        "//tensorflow/python/saved_model:pywrap_saved_model",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/python/training:py_checkpoint_reader",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:lazy_loader",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "checkpoint_test",
    srcs = ["checkpoint_test.py"],
    tags = [
        "no_pip",  # TODO(b/250108043)
        "no_windows",  # TODO(b/201457117)
    ],
    deps = [
        ":checkpoint",
        ":checkpoint_management",
        ":checkpoint_options",
        ":graph_view",
        ":save_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:stack",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training:checkpoint_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "checkpoint_context",
    srcs = [
        "checkpoint_context.py",
    ],
    srcs_version = "PY3",
)

tf_py_strict_test(
    name = "checkpoint_with_v1_optimizers_test",
    srcs = ["checkpoint_with_v1_optimizers_test.py"],
    deps = [
        ":checkpoint",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/trackable:autotrackable",
    ],
)

tf_py_strict_test(
    name = "checkpoint_metrics_test",
    srcs = ["checkpoint_metrics_test.py"],
    deps = [
        ":checkpoint",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:pywrap_saved_model",
    ],
)

py_strict_library(
    name = "checkpoint_view",
    srcs = ["checkpoint_view.py"],
    srcs_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":trackable_view",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training:py_checkpoint_reader",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "checkpoint_view_test",
    srcs = ["checkpoint_view_test.py"],
    tags = ["no_pip"],
    deps = [
        ":checkpoint_core",
        ":checkpoint_view",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/trackable:autotrackable",
    ],
)

py_strict_library(
    name = "graph_view",
    srcs = ["graph_view.py"],
    srcs_version = "PY3",
    deps = [
        ":save_util_v1",
        ":trackable_view",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "save_util",
    srcs = ["save_util.py"],
    srcs_version = "PY3",
    deps = [
        ":graph_view",
        ":save_util_v1",
        ":saveable_compat",
        ":util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:python_state",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/types",
        "//tensorflow/python/util:object_identity",
    ],
)

py_strict_library(
    name = "save_util_v1",
    srcs = ["save_util_v1.py"],
    srcs_version = "PY3",
    deps = [
        ":saveable_compat",
        ":util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:python_state",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:object_identity",
    ],
)

tf_py_strict_test(
    name = "save_util_v1_test",
    srcs = ["save_util_v1_test.py"],
    deps = [
        ":graph_view",
        ":save_util_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/util:object_identity",
    ],
)

py_strict_library(
    name = "trackable_view",
    srcs = ["trackable_view.py"],
    srcs_version = "PY3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:converter",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "trackable_view_test",
    srcs = ["trackable_view_test.py"],
    deps = [
        ":trackable_view",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/trackable:base",
    ],
)

py_strict_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/util:object_identity",
    ],
)

py_strict_library(
    name = "restore",
    srcs = ["restore.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_view",
        ":functional_saver",
        ":save_util_v1",
        ":saveable_compat",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:constants",
        "//tensorflow/python/trackable:python_state",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:object_identity",
    ],
)

tf_py_strict_test(
    name = "restore_test",
    srcs = ["restore_test.py"],
    deps = [
        ":checkpoint_core",
        ":restore",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/module",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training/saving:saveable_object",
    ],
)

tf_py_strict_test(
    name = "benchmarks_test",
    srcs = ["benchmarks_test.py"],
    deps = [
        ":checkpoint",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python/module",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training:py_checkpoint_reader",
    ],
)

tf_py_logged_benchmark(
    name = "benchmarks",
    target = "//tensorflow/python/checkpoint:benchmarks_test",
)

py_strict_library(
    name = "checkpoint_options",
    srcs = ["checkpoint_options.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "functional_saver",
    srcs = ["functional_saver.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_options",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
    ],
)

cuda_py_strict_test(
    name = "functional_saver_test",
    size = "medium",
    srcs = [
        "functional_saver_test.py",
    ],
    deps = [
        ":checkpoint_options",
        ":functional_saver",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/training/saving:saveable_object_util",
    ],
)

py_strict_library(
    name = "tensor_callable",
    srcs = ["tensor_callable.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/training/saving:saveable_object",
    ],
)

tf_py_strict_test(
    name = "tensor_callable_test",
    srcs = ["tensor_callable_test.py"],
    deps = [
        ":checkpoint",
        ":tensor_callable",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/trackable:base",
    ],
)

py_strict_library(
    name = "checkpoint_management",
    srcs = ["checkpoint_management.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:checkpoint_state_py",
        "//tensorflow/python/training:training_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "checkpoint_management_test",
    size = "small",
    srcs = [
        "checkpoint_management_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":checkpoint",
        ":checkpoint_management",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:checkpoint_state_py",
        "//tensorflow/python/training:saver",
    ],
)

py_strict_library(
    name = "saveable_compat",
    srcs = [
        "saveable_compat.py",
    ],
)

tf_py_strict_test(
    name = "saveable_compat_test",
    srcs = [
        "saveable_compat_test.py",
    ],
    data = [
        "testdata/table_legacy_saveable_object.data-00000-of-00001",
        "testdata/table_legacy_saveable_object.index",
    ],
    tags = ["no_pip"],
    deps = [
        ":checkpoint",
        ":generate_checkpoint_lib",
        ":saveable_compat",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training:checkpoint_utils",
        "//tensorflow/python/training/saving:saveable_object",
    ],
)

py_strict_binary(
    name = "generate_checkpoint",
    srcs = ["testdata/generate_checkpoint.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:checkpoint",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/module",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "generate_checkpoint_lib",
    srcs = ["testdata/generate_checkpoint.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:checkpoint",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/module",
        "@absl_py//absl:app",
    ],
)
