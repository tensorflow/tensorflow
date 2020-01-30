load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "py_test", "tf_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "profiler",
    srcs = ["profiler.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":model_analyzer",
        ":option_builder",
        ":tfprof_logger",
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "option_builder",
    srcs = ["option_builder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tfprof_logger",
    ],
)

py_library(
    name = "model_analyzer",
    srcs = ["model_analyzer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":option_builder",
        ":tfprof_logger",
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:_pywrap_tfprof",
        "//tensorflow/python:errors",
        "@six_archive//:six",
    ],
)

cuda_py_test(
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
        ":profile_context",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/profiler/internal:model_analyzer_testlib",
    ],
)

cuda_py_test(
    name = "profiler_test",
    srcs = ["profiler_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":model_analyzer",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/profiler/internal:model_analyzer_testlib",
    ],
)

py_library(
    name = "tfprof_logger",
    srcs = ["tfprof_logger.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python/profiler/internal:flops_registry",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "tfprof_logger_test",
    size = "small",
    srcs = ["tfprof_logger_test.py"],
    python_version = "PY3",
    deps = [
        ":tfprof_logger",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "profile_context",
    srcs = ["profile_context.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_analyzer",
        ":tfprof_logger",
    ],
)

cuda_py_test(
    name = "profile_context_test",
    srcs = ["profile_context_test.py"],
    python_version = "PY3",
    tags = [
        "no_gpu",  # b/136036359
        "no_pip",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":profile_context",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/profiler/internal:model_analyzer_testlib",
    ],
)

py_library(
    name = "pprof_profiler",
    srcs = ["pprof_profiler.py"],
    srcs_version = "PY2AND3",
    deps = ["@com_google_pprof//:pprof_proto_py"],
)

py_test(
    name = "pprof_profiler_test",
    size = "small",
    srcs = ["pprof_profiler_test.py"],
    main = "pprof_profiler_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = ["no_pip"],  # TODO(annarev): get it working with pip.
    deps = [
        ":pprof_profiler",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:math_ops",
        "@com_google_pprof//:pprof_proto_py",
    ],
)

py_test(
    name = "overview_page_proto_to_gviz_test",
    size = "small",
    srcs = ["overview_page_proto_to_gviz_test.py"],
    main = "overview_page_proto_to_gviz_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":overview_page_proto_to_gviz",
        "//tensorflow/core/profiler/protobuf:overview_page_proto_py_pb2",
        "//tensorflow/python:client_testlib",
        "//third_party/py/gviz_api",
    ],
)

py_test(
    name = "input_pipeline_proto_to_gviz_test",
    size = "small",
    srcs = ["input_pipeline_proto_to_gviz_test.py"],
    main = "input_pipeline_proto_to_gviz_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":input_pipeline_proto_to_gviz",
        "//tensorflow/core/profiler/protobuf:input_pipeline_proto_py_pb2",
        "//tensorflow/python:client_testlib",
        "//third_party/py/gviz_api",
    ],
)

py_library(
    name = "traceme",
    srcs = ["traceme.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/profiler/internal:_pywrap_traceme",
        "@six_archive//:six",
    ],
)

py_library(
    name = "scoped_annotation",
    srcs = ["scoped_annotation.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/profiler/internal:_pywrap_scoped_annotation",
        "@six_archive//:six",
    ],
)

py_library(
    name = "overview_page_proto_to_gviz",
    srcs = ["overview_page_proto_to_gviz.py"],
    srcs_version = "PY2AND3",
    deps = ["//third_party/py/gviz_api"],
)

py_library(
    name = "input_pipeline_proto_to_gviz",
    srcs = ["input_pipeline_proto_to_gviz.py"],
    srcs_version = "PY2AND3",
    deps = ["//third_party/py/gviz_api"],
)
