licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load("//tensorflow/core:platform/default/build_config.bzl", "pyx_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_python_default_plugins")

py_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_xla",
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
        "//tensorflow/compiler/xla:xla_data_proto_py",
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
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/client/lib:qr",
        "//tensorflow/compiler/xla/client/lib:self_adjoint_eig",
        "//tensorflow/compiler/xla/client/lib:svd",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service/cpu:custom_call_target_registry",
        "//tensorflow/core:lib",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "xrt",
    srcs = ["xrt.cc"],
    hdrs = ["xrt.h"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xrt:xrt_proto",
        "//tensorflow/compiler/xrt/cc:xrt_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

tf_py_wrap_cc(
    name = "pywrap_xla",
    srcs = [
        "xla.i",
    ],
    swig_includes = [
        "local_computation_builder.i",
        "xla_data.i",
        "//tensorflow/python:platform/base.i",
    ],
    version_script = select({
        "//tensorflow:macos": "pywrap_xla_exported_symbols.lds",
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
    ] + xla_python_default_plugins(),
)

tf_py_wrap_cc(
    name = "pywrap_xrt",
    srcs = [
        "xrt.i",
    ],
    swig_includes = [
        "xla_data.i",
        "//tensorflow/python:platform/base.i",
    ],
    version_script = select({
        "//tensorflow:macos": "pywrap_xla_exported_symbols.lds",
        "//tensorflow:windows": None,
        "//conditions:default": "pywrap_xla_version_script.lds",
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":numpy_bridge",
        ":xrt",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto",
    ],
)
