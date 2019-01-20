licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("//tensorflow/core:platform/default/build_config.bzl", "pyx_library")

py_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_xla",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/compiler/xla/service:hlo_proto_py",
    ],
)

pyx_library(
    name = "custom_call_for_test",
    testonly = True,
    srcs = ["custom_call_for_test.pyx"],
)

py_test(
    name = "xla_client_test",
    srcs = ["xla_client_test.py"],
    main = "xla_client_test.py",
    srcs_version = "PY2AND3",
    tags = ["no_oss"],
    deps = [
        ":custom_call_for_test",
        ":xla_client",
        "//tensorflow/python:platform_test",
    ],
)

cc_library(
    name = "numpy_bridge",
    srcs = ["numpy_bridge.cc"],
    hdrs = ["numpy_bridge.h"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/python:numpy_lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "local_computation_builder",
    srcs = ["local_computation_builder.cc"],
    hdrs = ["local_computation_builder.h"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:cholesky",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/client/lib:qr",
        "//tensorflow/compiler/xla/client/lib:triangular_solve",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service/cpu:custom_call_target_registry",
        "//tensorflow/compiler/xrt:xrt_proto",
        "//tensorflow/compiler/xrt/cc:xrt_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

tf_py_wrap_cc(
    name = "pywrap_xla",
    srcs = ["xla.i"],
    swig_includes = [
        "local_computation_builder.i",
        "//tensorflow/python:platform/base.i",
    ],
    version_script = select({
        "//tensorflow:darwin": "pywrap_xla_exported_symbols.lds",
        "//tensorflow:windows": None,
        "//conditions:default": "pywrap_xla_version_script.lds",
    }),
    deps = [
        ":local_computation_builder",
        ":numpy_bridge",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:cpu_plugin",
    ] + if_cuda_is_configured([
        "//tensorflow/compiler/xla/service:gpu_plugin",
    ]),
)
