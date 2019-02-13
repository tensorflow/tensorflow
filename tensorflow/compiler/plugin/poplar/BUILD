licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("@local_config_poplar//poplar:build_defs.bzl", "poplar_lib_directory")
load("@local_config_poplar//poplar:build_defs.bzl", "poplibs_lib_directory")
load("@local_config_poplar//poplar:build_defs.bzl", "tf_poplar_build_tag")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_proto_library")

# Rule for creating the vertex program pre-compiled object
genrule(
    name = "tf_graph_program",
    srcs = ["vertices/tf.cpp"],
    outs = ["tf.gp"],
    tools = ["@local_config_poplar//poplar:popc"],
    cmd = "$(location @local_config_poplar//poplar:popc) -DNDEBUG -O3 -I . -o $@ $<",
)

tf_proto_library(
    name = "protos",
    srcs = [
      "driver/trace.proto",
      "driver/executable.proto",
    ],
    cc_api_version = 2,
    default_header = True,
    j2objc_api_version = 1,
)

xla_proto_library(
    name = "backend_config",
    srcs = ["driver/backend_config.proto"],
    deps = ["//tensorflow/compiler/xla:xla_data_proto"],
)

cc_library(
    name = "custom_kernels_util",
    srcs = [
        "kernels/custom_kernels_util.cc",
    ],
    hdrs = [
        "kernels/custom_kernels_util.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:hlo",
        "@jsoncpp_git//:jsoncpp",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:bad_any_cast",
    ],
    alwayslink = True,
)

cc_library(
    name = "optimizers",
    srcs = [
        "driver/allocation_finder.cc",
        "driver/forward_allocation.cc",
        "driver/casts_elimination.cc",
        "driver/classification_predicates.cc",
        "driver/commutative_instruction_reorder_operands.cc",
        "driver/computation_flattener.cc",
        "driver/constant_slice_folding.cc",
        "driver/convolution_classifier.cc",
        "driver/expression_outliner.cc",
        "driver/find_all_users.cc",
        "driver/fuse_max_pool.cc",
        "driver/fuse_ops_early.cc",
        "driver/fuse_ops_late.cc",
        "driver/fuse_wide_const.cc",
        "driver/hlo_computation_name_uniquify.cc",
        "driver/hlo_hash.cc",
        "driver/hlo_matcher.cc",
        "driver/inplace_finder.cc",
        "driver/inplace_util.cc",
        "driver/input_output_aliasing_map.cc",
        "driver/matcher_predicates.cc",
        "driver/not_supported_gather_expander.cc",
        "driver/not_supported_scatter_expander.cc",
        "driver/root_token_replacer.cc",
        "driver/scheduler.cc",
        "driver/sharding_pass.cc",
        "driver/single_hlo_matcher.cc",
        "driver/util.cc",
        "driver/while_loop_condition_simplify.cc",
        "driver/while_loop_to_repeat_simplify.cc",
        "driver/while_loop_util.cc",
        "driver/wide_const_finder.cc",
    ],
    hdrs = [
        "driver/allocation_finder.h",
        "driver/forward_allocation.h",
        "driver/casts_elimination.h",
        "driver/classification_predicates.h",
        "driver/commutative_instruction_reorder_operands.h",
        "driver/compiler_annotations.h",
        "driver/computation_flattener.h",
        "driver/constant_slice_folding.h",
        "driver/convolution_classifier.h",
        "driver/expression_outliner.h",
        "driver/find_all_users.h",
        "driver/fuse_max_pool.h",
        "driver/fuse_ops_early.h",
        "driver/fuse_ops_late.h",
        "driver/fuse_wide_const.h",
        "driver/hlo_computation_name_uniquify.h",
        "driver/hlo_hash.h",
        "driver/hlo_matcher.h",
        "driver/inplace_finder.h",
        "driver/inplace_util.h",
        "driver/input_output_aliasing_map.h",
        "driver/matcher_predicates.h",
        "driver/meta_graph.h",
        "driver/not_supported_gather_expander.h",
        "driver/not_supported_scatter_expander.h",
        "driver/root_token_replacer.h",
        "driver/scheduler.h",
        "driver/sharding_pass.h",
        "driver/single_hlo_matcher.h",
        "driver/util.h",
        "driver/while_loop_condition_simplify.h",
        "driver/while_loop_to_repeat_simplify.h",
        "driver/while_loop_util.h",
        "driver/wide_const_finder.h",
    ],
    deps = [
        ":backend_config",
        ":custom_kernels_util",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:gather_expander",
        "//tensorflow/compiler/xla/service:scatter_expander",
        "//tensorflow/compiler/xla/service:while_loop_analysis",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf_headers",
    ],
)

cc_library(
    name = "common",
    srcs = [],
    hdrs = [
      "driver/xla_ipu_common.h",
    ],
)

cc_library(
    name = "driver",
    srcs = [
        "driver/norm_ops.cc",
        "driver/compiler.cc",
        "driver/conversions.cc",
        "driver/conv_op.cc",
        "driver/entry_visitor.cc",
        "driver/executable.cc",
        "driver/executor.cc",
        "driver/graph_caching_util.cc",
        "driver/norm_graph_caching.cc",
        "driver/conv_graph_caching.cc",
        "driver/map_ops.cc",
        "driver/math_ops.cc",
        "driver/platform.cc",
        "driver/platform_id.cc",
        "driver/poplar_util.cc",
        "driver/random_op.cc",
        "driver/reduction_ops.cc",
        "driver/sort_ops.cc",
        "driver/tensor.cc",
        "driver/tensor_ops.cc",
        "driver/transfer_manager.cc",
        "driver/visitor_base.cc",
        "driver/visitor_full.cc",
        "driver/visitor_inline_call.cc",
        "driver/visitor_arithmetic_expr.cc",
        "driver/visitor_map.cc",
        "driver/visitor_subcomputation.cc",
        "driver/custom_ops/custom_ops.cc",
        "driver/custom_ops/poplin_ops.cc",
        "driver/custom_ops/popnn_ops.cc",
        "driver/custom_ops/popops_ops.cc",
        "driver/custom_ops/poprand_ops.cc",
    ],
    hdrs = [
        "driver/compiler.h",
        "driver/compiler_resources.h",
        "driver/conversions.h",
        "driver/entry_visitor.h",
        "driver/executable.h",
        "driver/executor.h",
        "driver/graph_caching_util.h",
        "driver/norm_graph_caching.h",
        "driver/conv_graph_caching.h",
        "driver/ops.h",
        "driver/platform.h",
        "driver/platform_id.h",
        "driver/tensor.h",
        "driver/transfer_manager.h",
        "driver/vertex_templates.h",
        "driver/visitor_base.h",
        "driver/visitor_full.h",
        "driver/visitor_inline_call.h",
        "driver/visitor_arithmetic_expr.h",
        "driver/visitor_map.h",
        "driver/visitor_subcomputation.h",
        "driver/custom_ops/custom_ops.h",
        "driver/custom_ops/poplibs_ops.h",
        "driver/custom_ops/poplin_ops.h",
        "driver/custom_ops/popnn_ops.h",
        "driver/custom_ops/popops_ops.h",
        "driver/custom_ops/poprand_ops.h",
    ],
    copts = [
        "-DTF_POPLAR_BUILD_TAG=" + tf_poplar_build_tag(),
    ],
    deps = [
        ":common",
        ":custom_kernels_util",
        ":optimizers",
        ":protos_cc_impl",
        "//tensorflow/compiler/xla/service/cpu:cpu_runtime",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:dot_decomposer",
        "//tensorflow/compiler/xla/service:dynamic_index_splitter",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_get_dimension_size_rewriter",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_subcomputation_unification",
        "//tensorflow/compiler/xla/service:map_inliner",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:sort_simplifier",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_constant_sinking",
        "//tensorflow/compiler/xla/service:zero_sized_hlo_elimination",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@jsoncpp_git//:jsoncpp",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_poplar//poplar:poplar_headers",
        "@protobuf_archive//:protobuf_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "kernels",
    srcs = [
        "kernels/ipu_kernels.cc",
        "kernels/ipu_kernels_common.cc",
        "kernels/popnn_kernels.cc",
        "kernels/popops_kernels.cc",
        "kernels/pop_datastream_kernels.cc",
    ],
    hdrs = [
        "kernels/ipu_kernels_common.h",
    ],
    deps = [
        ":common",
        ":custom_kernels_util",
        ":driver",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//third_party/eigen3",
        "@jsoncpp_git//:jsoncpp",
        "@protobuf_archive//:protobuf_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "ipu_ops",
    srcs = ["ops/ipu_ops.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "ipu_ops_wrapper_py",
    out = "ops/gen_ipu_ops.py",
    deps = [":ipu_ops"],
)

tf_custom_op_py_library(
    name = "ipu_ops_py",
    kernels = [":ipu_ops"],
    deps = [":ipu_ops_wrapper_py"],
)

cc_library(
    name = "popnn_ops",
    srcs = ["ops/popnn_ops.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "popnn_ops_wrapper_py",
    out = "ops/gen_popnn_ops.py",
    deps = [":popnn_ops"],
    visibility = ["//tensorflow:internal"],
)

tf_custom_op_py_library(
    name = "popnn_ops_py",
    kernels = [":popnn_ops"],
    deps = [":popnn_ops_wrapper_py"],
    srcs_version = "PY2AND3",
)

cc_library(
    name = "pop_datastream_ops",
    srcs = ["ops/pop_datastream_ops.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "pop_datastream_ops_wrapper_py",
    out = "ops/gen_pop_datastream_ops.py",
    deps = [":pop_datastream_ops"],
    visibility = ["//tensorflow:internal"],
)

tf_custom_op_py_library(
    name = "pop_datastream_ops_py",
    kernels = [":pop_datastream_ops"],
    deps = [":pop_datastream_ops_wrapper_py"],
    srcs_version = "PY2AND3",
)


cc_library(
    name = "poplar_lib",
    srcs = [
        "driver/xla_ipu_device.cc",
    ],
    hdrs = [],
    deps = [
        ":driver",
        ":kernels",
        ":poplar_backend",
        ":ipu_ops",
        ":popnn_ops",
        ":pop_datastream_ops",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels:constant_op",
    ],
    linkopts = [
        "$(locations @local_config_poplar//poplar:poplar_lib)",
        "-Wl,-rpath," + poplar_lib_directory(),
        "-Wl,-rpath," + poplibs_lib_directory(),
    ],
    data = [
        ":tf_graph_program",
        ":protos_py",
        "@local_config_poplar//poplar:poplar_lib",
    ],
    linkstatic=1,
    alwayslink = True,
)

cc_library(
    name = "poplar_backend",
    srcs = [
        "driver/xla_ipu_backend.cc",
    ],
    hdrs = [],
    deps = [
        ":common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf_headers",
    ],
    linkstatic=1,
    alwayslink = True,
)

cc_library(
    name = "test_utils",
    srcs = [],
    hdrs = [
        "tests/test_utils.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:array3d",
    ],
)

tf_cc_test(
    name = "graph_compile_io_map_test",
    srcs = ["tests/graph_compile_io_map_test.cc"],
    extra_copts = [
        "-fexceptions",
    ],
    deps = [
        ":poplar_lib",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "conversions_test",
    srcs = ["tests/conversions_test.cc"],
    extra_copts = [
        "-fexceptions",
    ],
    deps = [
        ":poplar_lib",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "dynamic_slice_layout_test",
    srcs = ["tests/dynamic_slice_layout_test.cc"],
    extra_copts = [
        "-fexceptions",
    ],
    deps = [
        ":poplar_lib",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "sort_op_test",
    srcs = ["tests/sort_op_test.cc"],
    extra_copts = [
        "-fexceptions",
    ],
    deps = [
        ":poplar_lib",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "convolution_classifier_test",
    srcs = ["tests/convolution_classifier_test.cc"],
    size = "small",
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "inplace_test",
    srcs = ["tests/inplace_test.cc"],
    size = "small",
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "hlo_hash_test",
    srcs = ["tests/hlo_hash_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "hlo_matcher_test",
    srcs = ["tests/hlo_matcher_test.cc"],
    extra_copts = [
        "-fexceptions",
    ],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)


tf_cc_test(
    name = "expression_outliner_test",
    srcs = ["tests/expression_outliner_test.cc"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
    ],
)

tf_cc_test(
    name = "wide_const_finder_test",
    srcs = ["tests/wide_const_finder_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "while_loop_condition_simplify_test",
    srcs = ["tests/while_loop_condition_simplify_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "while_loop_to_repeat_simplify_test",
    srcs = ["tests/while_loop_to_repeat_simplify_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "commutative_instruction_reorder_operands_test",
    srcs = ["tests/commutative_instruction_reorder_operands_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "fuse_wide_const_test",
    srcs = ["tests/fuse_wide_const_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "scheduler_test",
    srcs = ["tests/scheduler_test.cc"],
    size = "small",
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "hlo_computation_name_uniquify_test",
    srcs = ["tests/hlo_computation_name_uniquify_test.cc"],
    size = "small",
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "fused_batch_norm_test",
    srcs = ["tests/fused_batch_norm_test.cc"],
    backends = [
        "poplar",
    ],
    size = "medium",
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "sharding_pass_test",
    srcs = ["tests/sharding_pass_test.cc"],
    size = "small",
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "constant_slice_folding_test",
    srcs = ["tests/constant_slice_folding_test.cc"],
    size = "small",
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "find_all_users_test",
    srcs = ["tests/find_all_users_test.cc"],
    extra_copts = [
        "-fexceptions",
    ],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "not_supported_gather_expander_test",
    srcs = ["tests/not_supported_gather_expander_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "not_supported_scatter_expander_test",
    srcs = ["tests/not_supported_scatter_expander_test.cc"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "root_token_replacer_test",
    srcs = ["tests/root_token_replacer_test.cc"],
    size = "small",
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

py_library(
    name = "test_utils_py",
    srcs = ["tests/test_utils.py"],
    deps = [
        ":ipu_ops_py",
        "//tensorflow/compiler/tf2xla/python:xla",
    ],
    srcs_version = "PY2AND3"
)

py_test(
    name = "device_test",
    size = "small",
    srcs = ["tests/device_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "ipu_model_device_test",
    size = "medium",
    srcs = ["tests/ipu_model_device_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "simple_network_test",
    size = "small",
    srcs = ["tests/simple_network_test.py"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "matmul_test",
    size = "small",
    srcs = ["tests/matmul_test.py"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "conv_test",
    size = "large",
    srcs = ["tests/conv_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:nn_ops",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "conv3d_test",
    size = "large",
    srcs = ["tests/conv3d_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:nn_ops",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "constant_test",
    size = "large",
    srcs = ["tests/constant_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:constant_op",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "variable_test",
    size = "small",
    srcs = ["tests/variable_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "multi_run_test",
    size = "small",
    srcs = ["tests/multi_run_test.py"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "reduce_test",
    size = "small",
    srcs = ["tests/reduce_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "f16_test",
    size = "small",
    srcs = ["tests/f16_test.py"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "64_bit_test",
    size = "small",
    srcs = ["tests/64_bit_test.py"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "batch_norm_test",
    size = "small",
    srcs = ["tests/batch_norm_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "fuse_ops_test",
    size = "medium",
    srcs = ["tests/fuse_ops_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "tensor_array_test",
    size = "small",
    srcs = ["tests/tensor_array_test.py"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "monitored_session_test",
    size = "small",
    srcs = ["tests/monitored_session_test.py"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:layers",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "estimator_test",
    size = "medium",
    srcs = ["tests/estimator_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:run_config",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python:layers",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "ipu_engine_cache_test",
    size = "medium",
    srcs = ["tests/ipu_engine_cache_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "casts_elimination_test",
    size = "medium",
    srcs = ["tests/casts_elimination_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "update_op_dependencies_test",
    size = "medium",
    srcs = ["tests/update_op_dependencies_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "norm_graph_caching_test",
    size = "medium",
    srcs = ["tests/norm_graph_caching_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "conv_graph_caching_test",
    size = "medium",
    srcs = ["tests/conv_graph_caching_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "lstm_test",
    size = "medium",
    srcs = ["tests/lstm_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ops/poplar:poplar"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "group_norm_test",
    size = "large",
    srcs = ["tests/group_norm_test.py"],
    shard_count = 5,
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ops/poplar:poplar"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "wide_const_expansion_test",
    size = "medium",
    srcs = ["tests/wide_const_expansion_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "popops_custom_ops_test",
    size = "medium",
    srcs = ["tests/popops_custom_ops_test.py"],
    deps = [
        ":test_utils_py",
        ":ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

test_suite(
    name = "all_tests",
    tests = [
        "device_test",
        "ipu_model_device_test",
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
        "hlo_hash_test",
        "hlo_matcher_test",
        "fuse_ops_test",
        "wide_const_finder_test",
        "monitored_session_test",
        "estimator_test",
        "expression_outliner_test",
        "convolution_classifier_test",
        "while_loop_condition_simplify_test",
        "conv3d_test",
        "ipu_engine_cache_test",
        "inplace_test",
        "commutative_instruction_reorder_operands_test",
        "casts_elimination_test",
        "update_op_dependencies_test",
        "fuse_wide_const_test",
        "norm_graph_caching_test",
        "conv_graph_caching_test",
        "scheduler_test",
        "sharding_pass_test",
        "constant_slice_folding_test",
        "dynamic_slice_layout_test",
        "sort_op_test",
        "while_loop_to_repeat_simplify_test",
        "lstm_test",
        "wide_const_expansion_test",
        "fused_batch_norm_test",
        "find_all_users_test",
        "hlo_computation_name_uniquify_test",
        "popops_custom_ops_test",
        "not_supported_gather_expander_test",
        "not_supported_scatter_expander_test",
        "group_norm_test",
        "root_token_replacer_test"
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
