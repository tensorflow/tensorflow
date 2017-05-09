licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

load("@local_config_poplar//poplar:build_defs.bzl", "poplar_lib_directory")
load("@local_config_poplar//poplar:build_defs.bzl", "popnn_lib_directory")

# Rule for creating the vertex program pre-compiled object
genrule(
    name = "tf_graph_program",
    srcs = ["vertices/tf.cpp"],
    outs = ["tf.gp"],
    tools = ["@local_config_poplar//poplar:popc"],
    cmd = "$(location @local_config_poplar//poplar:popc) -I . -o $@ $<",
)

cc_library(
    name = "poplar_lib",
    srcs = glob([
        "driver/*.cc",
        "stream_executor/*.cc",
    ]),
    hdrs = glob([
        "driver/*.h",
        "stream_executor/*.h",
    ]),
    deps = [
        "//third_party/eigen3",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        "@protobuf//:protobuf_headers",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_poplar//poplar:poplar_headers",
        "@local_config_poplar//popnn:popnn_headers",
    ],
    linkopts = [
        "$(locations @local_config_poplar//poplar:poplar_lib)",
        "$(locations @local_config_poplar//popnn:popnn_lib)",
        "-Wl,-rpath," + poplar_lib_directory(),
        "-Wl,-rpath," + popnn_lib_directory(),
    ],
    data = [
        ":tf_graph_program",
        "@local_config_poplar//poplar:poplar_lib",
        "@local_config_poplar//popnn:popnn_lib"
    ],
    linkstatic=0,
)

py_test(
    name = "device_test",
    size = "small",
    srcs = ["tests/device_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "simple_network_test",
    size = "small",
    srcs = ["tests/simple_network_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "matmul_test",
    size = "small",
    srcs = ["tests/matmul_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "conv_test",
    size = "large",
    srcs = ["tests/conv_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "variable_test",
    size = "small",
    srcs = ["tests/variable_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "multi_run_test",
    size = "small",
    srcs = ["tests/multi_run_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "reduce_test",
    size = "small",
    srcs = ["tests/reduce_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "device_test",
        "simple_network_test",
        "matmul_test",
        "multi_run_test",
        "conv_test",
        "variable_test",
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

