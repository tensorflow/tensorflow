licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")

py_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_xla",
        "//tensorflow/compiler/xla:xla_data_proto_py",
    ],
)

py_test(
    name = "xla_client_test",
    srcs = ["xla_client_test.py"],
    main = "xla_client_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":xla_client",
        "//tensorflow/python:platform_test",
    ],
)

cc_library(
    name = "numpy_bridge",
    srcs = ["numpy_bridge.cc"],
    hdrs = ["numpy_bridge.h"],
    deps = [
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/python:numpy_lib",
    ],
)

cc_library(
    name = "local_computation_builder",
    srcs = ["local_computation_builder.cc"],
    hdrs = ["local_computation_builder.h"],
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
    ],
)

tf_py_wrap_cc(
    name = "pywrap_xla",
    srcs = ["xla.i"],
    swig_includes = [
        "local_computation_builder.i",
    ],
    deps = [
        ":local_computation_builder",
        ":numpy_bridge",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:cpu_plugin",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
