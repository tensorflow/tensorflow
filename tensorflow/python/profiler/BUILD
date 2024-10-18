load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "profiler",
    srcs = ["profiler.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":model_analyzer",
        ":option_builder",
        ":tfprof_logger",
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "profiler_client",
    srcs = ["profiler_client.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/profiler/internal:_pywrap_profiler_plugin",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "profiler_client_test",
    srcs = ["profiler_client_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":profiler_client",
        ":profiler_v2",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "@pypi_portpicker//:pkg",
    ],
)

py_strict_library(
    name = "profiler_v2",
    srcs = ["profiler_v2.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler/internal:_pywrap_profiler",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "profiler_v2_test",
    srcs = ["profiler_v2_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
    ],
    deps = [
        ":profiler_v2",
        ":trace",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:gfile",
    ],
)

py_strict_test(
    name = "profiler_wrapper_test",
    srcs = ["profiler_wrapper_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
    ],
    deps = [
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/profiler/internal:_pywrap_profiler_plugin",
    ],
)

py_strict_library(
    name = "option_builder",
    srcs = ["option_builder.py"],
    srcs_version = "PY3",
    deps = [
        ":tfprof_logger",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "model_analyzer",
    srcs = ["model_analyzer.py"],
    srcs_version = "PY3",
    deps = [
        ":option_builder",
        ":tfprof_logger",
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:_pywrap_tfprof",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "model_analyzer_test",
    srcs = ["model_analyzer_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
        "notap",
        "oss_serial",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":model_analyzer",
        ":option_builder",
        ":profile_context",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/profiler/internal:model_analyzer_testlib",
        "//tensorflow/python/util:compat",
    ],
)

cuda_py_strict_test(
    name = "profiler_test",
    srcs = ["profiler_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":model_analyzer",
        ":option_builder",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/profiler/internal:model_analyzer_testlib",
    ],
)

py_strict_library(
    name = "tfprof_logger",
    srcs = ["tfprof_logger.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/profiler/internal:flops_registry",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "tfprof_logger_test",
    size = "small",
    srcs = ["tfprof_logger_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "profile_context",
    srcs = ["profile_context.py"],
    srcs_version = "PY3",
    deps = [
        ":model_analyzer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/util:_pywrap_tfprof",
        "//tensorflow/python/util:compat",
    ],
)

cuda_py_strict_test(
    name = "profile_context_test",
    srcs = ["profile_context_test.py"],
    python_version = "PY3",
    tags = [
        "no_gpu",  # b/136036359
        "no_pip",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":option_builder",
        ":profile_context",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/profiler/internal:model_analyzer_testlib",
    ],
)

py_strict_library(
    name = "pprof_profiler",
    srcs = ["pprof_profiler.py"],
    srcs_version = "PY3",
    deps = ["@com_google_pprof//:pprof_proto_py"],
)

py_strict_test(
    name = "pprof_profiler_test",
    size = "small",
    srcs = ["pprof_profiler_test.py"],
    main = "pprof_profiler_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_pip"],  # TODO(annarev): get it working with pip.
    deps = [
        ":pprof_profiler",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "@com_google_pprof//:pprof_proto_py",
    ],
)

py_strict_library(
    name = "trace",
    srcs = ["trace.py"],
    srcs_version = "PY3",
    visibility = [
        "//perftools/accelerators/xprof/xprofilez/integration_tests:__pkg__",
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/python/profiler/internal:_pywrap_traceme",
        "//tensorflow/python/util:tf_export",
    ],
)
