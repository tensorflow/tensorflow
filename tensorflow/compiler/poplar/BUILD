licenses(["restricted"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "poplar_plugin",
    srcs = [
        "driver/xla_ipu_device.cc",
    ],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/compiler/xla/service:local_service",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/jit:xla_cpu_device",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:all_kernels",
        "//tensorflow/stream_executor:stream_executor",
        "@local_config_poplar//poplar:poplar",
        "@local_config_poplar//popnn:popnn",
    ],
    linkstatic = 0,
)

genrule(
    name = "libpoplar_plugin",
    srcs = [":poplar_plugin"],
    outs = ["libpoplar_plugin.so"],
    cmd = "cp $(BINDIR)/tensorflow/compiler/poplar/libpoplar_plugin.so $(GENDIR)/tensorflow/compiler/poplar/libpoplar_plugin.so"
)

py_library(
    name = "poplar_plugin_py",
    srcs = [
        "poplar_plugin.py",
    ],
    data = [":libpoplar_plugin"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "device_test",
    size = "small",
    srcs = ["tests/device_test.py"],
    deps = [
        ":poplar_plugin_py",
        "//tensorflow:tensorflow_py"
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
)

