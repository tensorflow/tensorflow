licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

load("@local_config_poplar//poplar:build_defs.bzl", "poplar_lib_directory")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

# Rule for creating the vertex program pre-compiled object
genrule(
    name = "tf_graph_program",
    srcs = ["vertices/tf.cpp"],
    outs = ["tf.gp"],
    tools = ["@local_config_poplar//poplar:popc"],
    cmd = "$(location @local_config_poplar//poplar:popc) -I . -o $@ $<",
)

cc_library(
    name = "optimizers",
    srcs = [
        "driver/allocation_finder.cc",
        "driver/fuse_ops.cc",
        "driver/hlo_matcher.cc",
        "driver/inplace_finder.cc",
        "driver/matcher_predicates.cc",
        "driver/outliner.cc",
        "driver/scheduler.cc",
        "driver/util.cc",
        "driver/wide_const_finder.cc",
    ],
    hdrs = [
        "driver/allocation_finder.h",
        "driver/fuse_ops.h",
        "driver/hlo_matcher.h",
        "driver/inplace_finder.h",
        "driver/matcher_predicates.h",
        "driver/outliner.h",
        "driver/scheduler.h",
        "driver/util.h",
        "driver/wide_const_finder.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:hlo_scheduling",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf_headers",
    ],
)

cc_library(
    name = "driver",
    srcs = [
        "driver/compiler.cc",
        "driver/conversions.cc",
        "driver/conv_op.cc",
        "driver/executable.cc",
        "driver/executor.cc",
        "driver/map_ops.cc",
        "driver/math_ops.cc",
        "driver/op_util.cc",
        "driver/platform.cc",
        "driver/platform_id.cc",
        "driver/random_op.cc",
        "driver/reduction_ops.cc",
        "driver/tensor.cc",
        "driver/tensor_ops.cc",
        "driver/transfer_manager.cc",
        "driver/visitor_base.cc",
        "driver/visitor_full.cc",
        "driver/visitor_inline_call.cc",
        "driver/visitor_map.cc",
        "driver/visitor_subcomputation.cc",
    ],
    hdrs = [
        "driver/compiler.h",
        "driver/compiler_resources.h",
        "driver/conversions.h",
        "driver/executable.h",
        "driver/executor.h",
        "driver/ops.h",
        "driver/platform.h",
        "driver/platform_id.h",
        "driver/tensor.h",
        "driver/transfer_manager.h",
        "driver/vertex_templates.h",
        "driver/visitor_base.h",
        "driver/visitor_full.h",
        "driver/visitor_inline_call.h",
        "driver/visitor_map.h",
        "driver/visitor_subcomputation.h",
    ],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:batchnorm_expander",
        "//tensorflow/compiler/xla/service:flatten_call_graph",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:dot_decomposer",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_subcomputation_unification",
        "//tensorflow/compiler/xla/service:inliner",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf_headers",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_poplar//poplar:poplar_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "poplar_lib",
    srcs = [
        "driver/xla_ipu_device.cc",
    ],
    hdrs = [],
    deps = [
        ":driver",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        "//tensorflow/compiler/jit/kernels:xla_launch_op",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels:constant_op",
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
    alwayslink = True,
)

tf_cc_test(
    name = "graph_compile_io_map_test",
    srcs = ["tests/graph_compile_io_map_test.cc"],
    deps = [
        ":poplar_lib",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "conversions_test",
    srcs = ["tests/conversions_test.cc"],
    deps = [
        ":poplar_lib",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "outliner_test",
    srcs = ["tests/outliner_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "hlo_matcher_test",
    srcs = ["tests/hlo_matcher_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "allocation_finder_test",
    srcs = ["tests/allocation_finder_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "fuse_ops_test",
    srcs = ["tests/fuse_ops_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "wide_const_finder_test",
    srcs = ["tests/wide_const_finder_test.cc"],
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
    srcs_version = "PY2AND3",
)

py_test(
    name = "simple_network_test",
    size = "small",
    srcs = ["tests/simple_network_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "matmul_test",
    size = "small",
    srcs = ["tests/matmul_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "conv_test",
    size = "large",
    srcs = ["tests/conv_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "variable_test",
    size = "small",
    srcs = ["tests/variable_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "multi_run_test",
    size = "small",
    srcs = ["tests/multi_run_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "reduce_test",
    size = "small",
    srcs = ["tests/reduce_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "f16_test",
    size = "small",
    srcs = ["tests/f16_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "64_bit_test",
    size = "small",
    srcs = ["tests/64_bit_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "batch_norm_test",
    size = "small",
    srcs = ["tests/batch_norm_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "tensor_array_test",
    size = "small",
    srcs = ["tests/tensor_array_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "monitored_session_test",
    size = "small",
    srcs = ["tests/monitored_session_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
    srcs_version = "PY2AND3",
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
        "tensor_array_test",
        "fuse_ops_test",
        "outliner_test",
        "hlo_matcher_test",
        "wide_const_finder_test",
        "monitored_session_test",
    ],
)

test_suite(
    name = "all_ci_tests",
    tests = [
        "device_test",
        "simple_network_test",
        "matmul_test",
        "multi_run_test",
        "conv_test",
        "variable_test",
        "reduce_test",
        "f16_test",
        "64_bit_test",
        "allocation_finder_test",
        "batch_norm_test",
        "fuse_ops_test",
        "hlo_matcher_test",
        "wide_const_finder_test",
        "monitored_session_test",
    ],
)

exports_files([
    "disabled_manifest.txt",
    "disables_xla_tests_manifest.txt",
])

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

