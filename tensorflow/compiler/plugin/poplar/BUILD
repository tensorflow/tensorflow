licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

load("@local_config_poplar//poplar:build_defs.bzl", "poplar_lib_directory")

# Rule for creating the vertex program pre-compiled object
genrule(
    name = "tf_graph_program",
    srcs = ["vertices/tf.cpp"],
    outs = ["tf.gp"],
    tools = ["@local_config_poplar//poplar:popc"],
    cmd = "$(location @local_config_poplar//poplar:popc) -I . -o $@ $<",
)

cc_library(
    name = "driver",
    srcs = [
        "driver/compiler.cc",
        "driver/conversions.cc",
        "driver/conv_op.cc",
        "driver/executable.cc",
        "driver/map_ops.cc",
        "driver/math_ops.cc",
        "driver/op_util.cc",
        "driver/random_op.cc",
        "driver/reduction_ops.cc",
        "driver/tensor.cc",
        "driver/tensor_ops.cc",
        "driver/transfer_manager.cc",
        "driver/visitor_base.cc",
        "driver/visitor_call.cc",
        "driver/visitor_full.cc",
        "driver/visitor_map.cc",
    ],
    hdrs = [
        "driver/compiler.h",
        "driver/compiler_resources.h",
        "driver/conversions.h",
        "driver/executable.h",
        "driver/ops.h",
        "driver/tensor.h",
        "driver/transfer_manager.h",
        "driver/vertex_templates.h",
        "driver/visitor_base.h",
        "driver/visitor_call.h",
        "driver/visitor_full.h",
        "driver/visitor_map.h",
    ],
    deps = [
        ":executor",
        ":optimizers",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf_headers",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_poplar//poplar:poplar_headers",
    ],
)

cc_library(
    name = "executor",
    srcs = [
        "stream_executor/executor.cc",
        "stream_executor/platform.cc",
        "stream_executor/platform_id.cc",
    ],
    hdrs = [
        "stream_executor/executor.h",
        "stream_executor/platform.h",
        "stream_executor/platform_id.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_headers",
    ],
)

cc_library(
    name = "optimizers",
    srcs = [
        "driver/allocation_finder.cc",
        "driver/fuse_ops.cc",
        "driver/outliner.cc",
    ],
    hdrs = [
        "driver/allocation_finder.h",
        "driver/fuse_ops.h",
        "driver/outliner.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf_headers",
    ],
)

cc_library(
    name = "poplar_lib",
    srcs = [
        "driver/xla_ipu_device.cc",
    ],
    hdrs = [],
    deps = [
        ":driver",
        ":executor",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
    ],
    linkopts = [
        "$(locations @local_config_poplar//poplar:poplar_lib)",
        "-Wl,-rpath," + poplar_lib_directory(),
    ],
    data = [
        ":tf_graph_program",
        "@local_config_poplar//poplar:poplar_lib",
    ],
    linkstatic=1,
)

cc_test(
    name = "graph_compile_io_map_test",
    srcs = ["tests/graph_compile_io_map_test.cc"],
    deps = [
        ":poplar_lib",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "conversions_test",
    srcs = ["tests/conversions_test.cc"],
    deps = [
        ":poplar_lib",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "outliner_test",
    srcs = ["tests/outliner_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "allocation_finder_test",
    srcs = ["tests/allocation_finder_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "fuse_ops_test",
    srcs = ["tests/fuse_ops_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
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

py_test(
    name = "f16_test",
    size = "small",
    srcs = ["tests/f16_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "64_bit_test",
    size = "small",
    srcs = ["tests/64_bit_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "batch_norm_test",
    size = "small",
    srcs = ["tests/batch_norm_test.py"],
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
        "reduce_test",
        "f16_test",
        "graph_compile_io_map_test",
        "conversions_test",
        "64_bit_test",
        "allocation_finder_test",
        "batch_norm_test",
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

