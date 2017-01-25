licenses(["restricted"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar", "pkg_deb")

package(default_visibility = ["//visibility:public"])

# Package the plugin into a tar file (TODO: make a pip file)
pkg_tar(
    name = "poplar",
    files = [
        "libpoplar_plugin.so",
        "poplar_plugin.py",
        "tf.gp",
    ],
)

# Rule for creating the vertex program pre-compiled object
genrule(
    name = "tf_graph_program",
    srcs = ["vertices/tf.cpp"],
    outs = ["tf.gp"],
    tools = ["@local_config_poplar//poplar:popc"],
    cmd = "$(location @local_config_poplar//poplar:popc) -I . -o $@ $<",
)

# Ideally this rule would be a cc_binary.  It cannot be changed until the bulk
# of Tensorflow is available as traditional external .h files and a .so file
cc_library(
    name = "poplar_lib",
    srcs = glob([
        "driver/*.cc",
        "stream_executor/*.cc",
    ]) + [
        "@local_config_poplar//poplar:poplar_lib",
        "@local_config_poplar//popnn:popnn_lib",
    ],
    hdrs = glob([
        "driver/*.h",
        "stream_executor/*.h",
    ]),
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/compiler/xla/service:local_service",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/jit:xla_cpu_device",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:all_kernels",
        "//tensorflow/stream_executor:stream_executor",
        "@local_config_poplar//poplar:poplar",
        "@local_config_poplar//popnn:popnn",
    ],
    linkopts = [
        "@local_config_poplar//poplar:poplar_lib",
        "@local_config_poplar//popnn:popnn_lib",
    ],
    linkstatic = 0,
)

# This patches up the rpaths of the .so file so that it can be loaded by
# the test harness without failing to find libpoplar.so and libpopnn.so
genrule(
    name = "libpoplar_plugin",
    srcs = [
        ":poplar_lib",
        "@local_config_poplar//poplar:poplar_lib",
        "@local_config_poplar//popnn:popnn_lib",
    ],
    tools = [
        "apply_poplar_rpaths.py"
    ],
    outs = ["libpoplar_plugin.so"],
    cmd = "python $(location apply_poplar_rpaths.py) " +
          "$(BINDIR)/tensorflow/compiler/poplar/libpoplar_lib.so " +
          "$(GENDIR)/tensorflow/compiler/poplar/libpoplar_plugin.so " +
          "$(location @local_config_poplar//poplar:poplar_lib) " +
          "$(location @local_config_poplar//popnn:popnn_lib)"
)

py_library(
    name = "poplar_plugin_py",
    srcs = [
        "poplar_plugin.py",
    ],
    data = [
        ":libpoplar_plugin",
        ":tf_graph_program",
    ],
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
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "simple_network_test",
    size = "small",
    srcs = ["tests/simple_network_test.py"],
    deps = [
        ":poplar_plugin_py",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "matmul_test",
    size = "small",
    srcs = ["tests/matmul_test.py"],
    deps = [
        ":poplar_plugin_py",
        "//tensorflow:tensorflow_py",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "device_test",
        "simple_network_test",
        "matmul_test",
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

