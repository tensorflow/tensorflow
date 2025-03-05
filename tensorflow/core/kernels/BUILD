load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm",
    "if_rocm_hipblaslt",
)
load(
    "@local_xla//xla/tsl/mkl:build_defs.bzl",
    "if_mkl",
    "mkl_deps",
)
load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "if_cuda_or_rocm",
    "if_google",
    "if_mobile",
    "if_nccl",
    "if_not_windows",
    "if_oss",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_copts",
    "tf_cuda_library",
    "tf_cuda_only_cc_test",
    "tf_opts_nortti_if_lite_protos",
)
load("//tensorflow:tensorflow.default.bzl", "cc_header_only_library", "filegroup", "get_compatible_with_portable", "tf_cc_shared_library", "tf_cuda_cc_test", "tf_cuda_cc_tests", "tf_disable_ptxas_warning_flags", "tf_kernel_library")
load(
    "//tensorflow/core/kernels/mlir_generated:build_defs.bzl",
    "if_mlir_generated_experimental_kernels_enabled",
    "if_mlir_generated_gpu_kernels_enabled",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_fingerprint_deps",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# Description:
# Op kernel implementations for TensorFlow.
#
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    features = if_google(
        [
            "-parse_headers",
        ],
    ),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/...",
        "//tensorflow_text/...",
        "//waymo/onboard/ml/...",
    ],
)

package_group(
    name = "optimizer_helper_friends",
    packages = [
        "//learning/brain/research/lather/...",
        "//learning/clair/alise/...",
    ],
)

# TODO(b/315862822): Deprecate the usage of `--define=tensorflow_mkldnn_contraction_kernel=0` flag
config_setting(
    name = "defines_no_mkldnn_contraction_kernel",
    define_values = {
        "tensorflow_mkldnn_contraction_kernel": "0",
    },
)

bool_flag(
    name = "disable_mkldnn_contraction_kernel",
    build_setting_default = False,
)

config_setting(
    name = "flag_no_mkldnn_contraction_kernel",
    flag_values = {
        ":disable_mkldnn_contraction_kernel": "True",
    },
)

selects.config_setting_group(
    # Add "--define tensorflow_mkldnn_contraction_kernel=0" (not preferable) or build flag
    # "//tensorflow/core/kernels:disable_mkldnn_contraction_kernel=True" (preferred)
    # to your build command to disable mkldnn sgemm in
    # Eigen tensor contractions (matrix multiplications and convolutions). The mkldnn
    # kernels are generated at runtime and use avx/avx2/fma/avx512 based on cpu status registers
    # (https://en.wikipedia.org/wiki/CPUID). Default Eigen contraction kernel is
    # Eigen::internal::gebp_kernel (general block-panel kernel).
    name = "no_mkldnn_contraction_kernel",
    match_any = [
        ":defines_no_mkldnn_contraction_kernel",
        ":flag_no_mkldnn_contraction_kernel",
    ],
)

# Public support libraries ----------------------------------------------------

cc_library(
    name = "assign_op",
    hdrs = ["assign_op.h"],
    deps = [
        "//tensorflow/core:framework",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "strided_slice_op",
    srcs = [
        "strided_slice_op.cc",
        "strided_slice_op_inst_0.cc",
        "strided_slice_op_inst_1.cc",
        "strided_slice_op_inst_2.cc",
        "strided_slice_op_inst_3.cc",
        "strided_slice_op_inst_4.cc",
        "strided_slice_op_inst_5.cc",
        "strided_slice_op_inst_6.cc",
        "strided_slice_op_inst_7.cc",
        "strided_slice_op_inst_8.cc",
    ],
    hdrs = [
        "slice_op.h",
        "strided_slice_op.h",
    ],
    gpu_srcs = [
        "slice_op.h",
        "strided_slice_op.h",
        "strided_slice_op_impl.h",
        "strided_slice_op_gpu_impl.h",
        "strided_slice_op_gpu_int.cu.cc",
        "strided_slice_op_gpu_complex.cu.cc",
        "strided_slice_op_gpu_bool.cu.cc",
        "strided_slice_op_gpu_number_types.cu.cc",
    ],
    textual_hdrs = ["strided_slice_op_impl.h"],
    deps = [
        ":dense_update_functor",
        ":inplace_ops",
        ":ops_util",
        ":training_op_helpers",
        ":variable_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:bounds_check",
        "@com_google_absl//absl/base:prefetch",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "clustering_ops",
    prefix = "clustering_ops",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "clustering_ops_test",
    srcs = ["clustering_ops_test.cc"],
    deps = [
        ":clustering_ops",
        "//tensorflow/core:clustering_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "collective_ops",
    srcs = if_nccl([
        "collective_nccl.h",
        "collective_nccl.cc",
        "collective_nccl_all_to_all.h",
        "collective_nccl_all_to_all.cc",
        "collective_nccl_broadcaster.h",
        "collective_nccl_broadcaster.cc",
        "collective_nccl_gatherer.h",
        "collective_nccl_gatherer.cc",
        "collective_nccl_reducer.h",
        "collective_nccl_reducer.cc",
    ]),
    features = ["-layering_check"],
    prefix = "collective_ops",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/activity_watcher",
        "//tensorflow/core/activity_watcher:activity_watcher_utils",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme",
    ] + if_nccl([
        "//tensorflow/core/nccl:collective_communicator",
    ]),
)

tf_cuda_cc_test(
    name = "collective_nccl_test",
    size = "small",
    srcs = ["collective_nccl_test.cc"],
    features = if_cuda(["-layering_check"]),
    tags = tf_cuda_tests_tags() + [
        "guitar",
        "multi_gpu",
        "no_oss",
        "notap",  # TODO(b/287692888): re-enable once the tests passes.
    ],
    deps = [
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:collective_test_util",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/nccl:collective_communicator",
    ],
)

tf_kernel_library(
    name = "concat_lib",
    srcs = [
        "concat_lib_cpu.cc",
        "concat_lib_gpu.cc",
    ],
    hdrs = [
        "concat_lib.h",
        "concat_lib_cpu.h",
    ],
    gpu_copts = if_not_windows([
        "-Wno-pass-failed",  # clang misses #pragma loop optimizations
    ]),
    gpu_srcs = [
        "concat_lib_gpu_impl.cu.cc",
        "concat_lib.h",
        "concat_lib_gpu.h",
        "gpu_device_array.h",
        "gpu_device_array_gpu.h",
    ],
    deps = [
        ":loose_headers",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/framework:bounds_check",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 0,
)

cc_library(
    name = "concat_lib_hdrs",
    hdrs = [
        "concat_lib.h",
        "concat_lib_cpu.h",
    ],
    deps = ["@eigen_archive//:eigen3"],
)

tf_kernel_library(
    name = "conv_2d",
    hdrs = ["conv_2d.h"],
    features = if_cuda(["-layering_check"]),
    gpu_copts = if_not_windows([
        "-Wno-pass-failed",  # clang misses #pragma loop optimizations
    ]),
    gpu_srcs = [
        "conv_2d.h",
        "conv_2d_gpu.h",
        "conv_2d_gpu_double.cu.cc",
        "conv_2d_gpu_float.cu.cc",
        "conv_2d_gpu_half.cu.cc",
        "conv_2d_gpu_bfloat16.cu.cc",
        "conv_2d_gpu_int.cu.cc",
        "conv_2d_gpu_int_spatial_convolution.cu.cc",
        "conv_2d_gpu_int_spatial_convolution_backward.cu.cc",
        "conv_2d_gpu_uint16.cu.cc",
        "conv_2d_gpu_uint32.cu.cc",
        "conv_2d_gpu_uint64.cu.cc",
        "conv_2d_gpu_uint8.cu.cc",
    ],
    deps = [
        ":eigen_helpers",
        ":fill_functor",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ] + if_cuda_or_rocm([":gpu_utils"]),
    alwayslink = 1,
)

cc_library(
    name = "conv_2d_hdrs",
    hdrs = ["conv_2d.h"],
    deps = [
        ":eigen_helpers",
        "//tensorflow/core/framework:bounds_check",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "conv_3d",
    hdrs = ["conv_3d.h"],
    deps = [
        ":eigen_helpers",
        "//tensorflow/core:framework",
    ],
)

tf_kernel_library(
    name = "fill_functor",
    features = ["-layering_check"],
    prefix = "fill_functor",
    deps = [
        "//tensorflow/core:framework",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "initializable_lookup_table",
    srcs = ["initializable_lookup_table.cc"],
    hdrs = ["initializable_lookup_table.h"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "lookup_util",
    srcs = ["lookup_util.cc"],
    hdrs = ["lookup_util.h"],
    deps = [
        ":initializable_lookup_table",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:op_requires",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "nccl_kernels",
    srcs = if_cuda_or_rocm([
        "nccl_ops.cc",
    ]),
    deps = if_cuda([
        "@local_config_nccl//:nccl",
    ]) + if_rocm([
        "@local_config_rocm//rocm:rccl",
    ]) + if_cuda_or_rocm([
        "//tensorflow/core/nccl:nccl_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
    ]),
)

cc_library(
    name = "sparse_utils",
    srcs = [
        "sparse_utils.cc",
    ],
    hdrs = ["sparse_utils.h"],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "sparse_utils_test",
    srcs = ["sparse_utils_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":sparse_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status_matchers",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "tensor_flag_utils",
    srcs = [
        "tensor_flag_utils.cc",
    ],
    hdrs = ["tensor_flag_utils.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "tensor_flag_utils_test",
    srcs = ["tensor_flag_utils_test.cc"],
    deps = [
        ":tensor_flag_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/base:core_headers",
    ],
)

tf_cuda_library(
    name = "ops_testutil",
    testonly = 1,
    srcs = ["ops_testutil.cc"],
    hdrs = ["ops_testutil.h"],
    cuda_deps = [
        "//tensorflow/core:gpu_lib",
        "//tensorflow/core:gpu_runtime",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:tensor_testutil",
    ],
)

cc_library(
    name = "ops_util",
    hdrs = ["ops_util.h"],
    copts = if_not_windows(["-Wno-sign-compare"]),
    deps = [
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "ops_util_hdrs",
    hdrs = ["ops_util.h"],
    deps = ["@eigen_archive//:eigen3"],
)

cc_library(
    name = "gpu_device_array",
    hdrs = [
        "gpu_device_array.h",
        "gpu_device_array_gpu.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "gpu_prim_hdrs",
    hdrs = ["gpu_prim.h"],
    deps = if_cuda([
        "@local_config_cuda//cuda:cub_headers",
    ]) + if_rocm([
        "@local_config_rocm//rocm:rocprim",
    ]),
)

cc_library(
    name = "gpu_prim_helpers",
    hdrs = ["gpu_prim_helpers.h"],
    deps = if_cuda_or_rocm([
        ":gpu_prim_hdrs",
    ]),
)

tf_cuda_only_cc_test(
    name = "gpu_prim_helpers_test",
    srcs = ["gpu_prim_helpers_test.cu.cc"],
    features = if_cuda(["-layering_check"]),
    tags = ["no_cuda_asan"],  # TODO(b/183963619)
    deps = [
        ":gpu_prim_helpers",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "conv_ops_gpu_hdrs",
    hdrs = ["conv_ops_gpu.h"],
)

# We keep this target only because some contrib/ targets depend on it. The
# reason why the contrib/ targets can't depend on gpu_utils is that, some
# of the targets are tf_custom_op_library. tf_custom_op_library forbids the
# dependency to tensorflow/core:lib, which gpu_utils certainly depends on.
cc_library(
    name = "gpu_util_hdrs",
    hdrs = [
        "gpu_utils.h",
    ],
    deps = ["@local_xla//xla/stream_executor:lazy_op_runner"],
)

cc_library(
    name = "numeric_options_utils",
    hdrs = ["numeric_options_utils.h"],
    deps = [
        "//tensorflow/core/platform:tensor_float_32_hdr_lib",
        "//tensorflow/core/util:determinism_for_kernels",
        "//tensorflow/core/util:env_var",
        "@local_xla//xla/stream_executor:numeric_options",
        "@local_xla//xla/tsl/util:determinism_for_kernels",
    ] + if_static(["//tensorflow/core/platform:tensor_float_32_utils"]),
)

tf_cuda_library(
    name = "gpu_utils",
    srcs = if_cuda_or_rocm(["gpu_utils.cc"]),
    hdrs = ["gpu_utils.h"],
    features = ["-layering_check"],
    deps = [
        ":gpu_util_hdrs",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/protobuf:autotuning_proto_cc",
        "//tensorflow/core/protobuf:conv_autotuning_proto_cc",
        "//tensorflow/core/util:determinism_for_kernels",
        "//tensorflow/core/util:env_var",
        "//tensorflow/core/util/autotune_maps:conv_parameters",
        "//tensorflow/core/util/proto:proto_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/stream_executor:lazy_op_runner",
        "@local_xla//xla/stream_executor/gpu:asm_compiler",
        "@local_xla//xla/stream_executor/gpu:redzone_allocator",
    ] + if_cuda([
        "@local_config_cuda//cuda:cudnn_header",
        "@local_xla//xla/stream_executor/integrations:tf_allocator_adapter",
    ]),
)

tf_cc_test(
    name = "ops_util_test",
    size = "small",
    srcs = ["ops_util_test.cc"],
    deps = [
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "reshape_util",
    srcs = ["reshape_util.cc"],
    hdrs = ["reshape_util.h"],
    gpu_srcs = [
        "reshape_util_gpu.cu.cc",
        "reshape_util.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "variable_ops_test",
    size = "small",
    srcs = ["variable_ops_test.cc"],
    deps = [
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:direct_session_internal",
    ],
)

tf_kernel_library(
    name = "stage_op",
    srcs = ["stage_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "map_stage_op",
    srcs = ["map_stage_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "queue_base",
    srcs = ["queue_base.cc"],
    hdrs = ["queue_base.h"],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "queue_op",
    srcs = ["queue_op.cc"],
    hdrs = ["queue_op.h"],
    features = ["-layering_check"],
    deps = [
        ":queue_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "priority_queue",
    srcs = ["priority_queue.cc"],
    hdrs = ["priority_queue.h"],
    deps = [
        ":queue_base",
        ":typed_queue",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "batch_kernels",
    srcs = ["batch_kernels.cc"],
    hdrs = ["batch_kernels.h"],
    deps = [
        ":ops_util_hdrs",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/batching_util:adaptive_shared_batch_scheduler",
        "//tensorflow/core/kernels/batching_util:batch_resource_base",
        "//tensorflow/core/kernels/batching_util:batch_scheduler_hdrs",
        "//tensorflow/core/kernels/batching_util:batch_scheduler_utils",
        "//tensorflow/core/kernels/batching_util:bounded_executor",
        "//tensorflow/core/kernels/batching_util:concat_split_util",
        "//tensorflow/core/kernels/batching_util:periodic_function_dynamic",
        "//tensorflow/core/kernels/batching_util:warmup",
        "//tensorflow/core/platform:numbers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla/tsl/platform:types",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "record_input_op",
    srcs = [
        "record_input_op.cc",
        "record_yielder.cc",
        "record_yielder.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "save_restore_tensor",
    srcs = ["save_restore_tensor.cc"],
    hdrs = ["save_restore_tensor.h"],
    copts = if_not_windows(["-Wno-sign-compare"]),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

tf_kernel_library(
    name = "split_lib",
    srcs = ["split_lib_cpu.cc"],
    hdrs = ["split_lib.h"],
    gpu_srcs = [
        "split_lib_gpu.cu.cc",
        "split_lib.h",
        "split_lib_gpu.h",
    ],
    deps = [
        ":gpu_device_array",
        "//tensorflow/core:framework",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 0,
)

cc_library(
    name = "split_lib_hdrs",
    hdrs = ["split_lib.h"],
    deps = [
        "//tensorflow/core:framework_lite",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "typed_queue",
    hdrs = ["typed_queue.h"],
    deps = [
        ":queue_base",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "training_op_helpers",
    srcs = ["training_op_helpers.cc"],
    hdrs = ["training_op_helpers.h"],
    visibility = [
        ":friends",
        ":optimizer_helper_friends",
    ],
    deps = [
        ":dense_update_functor",
        ":variable_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:mutex",
    ],
)

# Private support libraries ---------------------------------------------------

filegroup(
    name = "xla_cpu_runtime_hdrs",
    srcs = [
        "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel.h",
        "@local_xla//xla/tsl/framework/convolution:eigen_convolution_helpers.h",
        "@local_xla//xla/tsl/framework/convolution:eigen_spatial_convolutions.h",
        "@local_xla//xla/tsl/framework/convolution:eigen_spatial_convolutions-inl.h",
    ],
)

filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = [
        "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel.cc",
    ],
)

cc_library(
    name = "redux_functor",
    hdrs = ["redux_functor.h"],
    deps = [
        "//tensorflow/core:framework",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "fused_eigen_output_kernels",
    srcs = ["fused_eigen_output_kernels.cc"],
    hdrs = ["fused_eigen_output_kernels.h"],
    deps = [
        "//tensorflow/core:framework",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "eigen_helpers",
    hdrs = [
        "eigen_activations.h",
        "eigen_attention.h",
        "eigen_backward_cuboid_convolutions.h",
        "eigen_backward_spatial_convolutions.h",
        "eigen_cuboid_convolution.h",
        "eigen_pooling.h",
        "@local_xla//xla/tsl/framework/convolution:eigen_spatial_convolutions.h",
    ],
    compatible_with = get_compatible_with_portable(),
    defines = ["EIGEN_NEON_GEBP_NR=4"],
    deps = [
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel",
        "@local_xla//xla/tsl/framework/convolution:eigen_convolution_helpers",
        "@local_xla//xla/tsl/framework/convolution:eigen_helpers",
        "@local_xla//xla/tsl/framework/convolution:eigen_spatial_convolutions-inl",
    ],
)

cc_library(
    name = "eigen_helpers_no_mkl",
    hdrs = [
        "eigen_activations.h",
        "eigen_attention.h",
        "eigen_backward_cuboid_convolutions.h",
        "eigen_backward_spatial_convolutions.h",
        "eigen_cuboid_convolution.h",
        "eigen_pooling.h",
    ],
    defines = ["EIGEN_NEON_GEBP_NR=4"],
    deps = [
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/framework/convolution:eigen_convolution_helpers",
        "@local_xla//xla/tsl/framework/convolution:eigen_spatial_convolutions-inl",
    ],
)

# OpKernel libraries ----------------------------------------------------------

ARRAY_DEPS = [
    ":concat_lib",
    ":fill_functor",
    ":gather_functor",
    ":ops_util",
    ":transpose_functor",
    "@com_google_absl//absl/base:prefetch",
    "//tensorflow/core:array_grad",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core/framework:bounds_check",
    "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
    "//tensorflow/core/util:bad_indices_policy",
    "@eigen_archive//:eigen3",
]

tf_kernel_library(
    name = "immutable_constant_op",
    prefix = "immutable_constant_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "set_kernels",
    prefix = "set_kernels",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "debug_ops",
    features = ["-layering_check"],
    prefix = "debug_ops",
    deps = ARRAY_DEPS + [
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core/debug:debug_io_utils",
    ],
)

cc_library(
    name = "array",
    deps = [
        ":batch_space_ops",
        ":bcast_ops",
        ":broadcast_to_op",
        ":concat_op",
        ":constant_op",
        ":depth_space_ops",
        ":diag_op",
        ":edit_distance_op",
        ":fingerprint_op",
        ":gather_nd_op",
        ":gather_op",
        ":guarantee_const_op",
        ":host_constant_op",
        ":identity_n_op",
        ":identity_op",
        ":immutable_constant_op",
        ":inplace_ops",
        ":listdiff_op",
        ":one_hot_op",
        ":pack_op",
        ":pad_op",
        ":quantize_and_dequantize_op",
        ":reshape_op",
        ":reverse_op",
        ":reverse_sequence_op",
        ":searchsorted_op",
        ":shape_ops",
        ":slice_op",
        ":snapshot_op",
        ":split_op",
        ":split_v_op",
        ":strided_slice_op",
        ":tile_ops",
        ":transpose_op",
        ":unique_op",
        ":unpack_op",
        ":unravel_index_op",
        ":where_op",
    ],
)

tf_kernel_library(
    name = "bcast_ops",
    prefix = "bcast_ops",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "broadcast_to_op",
    prefix = "broadcast_to_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "concat_op",
    features = ["-layering_check"],
    prefix = "concat_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "guarantee_const_op",
    prefix = "guarantee_const_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "constant_op",
    copts = if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    # *.cu.cc sources are compiled with gpu_copts instead of copts.
    gpu_copts = if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    prefix = "constant_op",
    deps = ARRAY_DEPS + [
        "//tensorflow/core/kernels/mlir_generated:constant_op",
    ],
)

tf_kernel_library(
    name = "host_constant_op",
    prefix = "host_constant_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "diag_op",
    prefix = "diag_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "edit_distance_op",
    prefix = "edit_distance_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "fingerprint_op",
    prefix = "fingerprint_op",
    deps = ARRAY_DEPS,
)

tf_cc_test(
    name = "fingerprint_op_test",
    size = "small",
    srcs = ["fingerprint_op_test.cc"],
    deps = [
        ":fingerprint_op",
        ":ops_testutil",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "gather_nd_op",
    prefix = "gather_nd_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "gather_op",
    prefix = "gather_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "identity_op",
    prefix = "identity_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "identity_n_op",
    prefix = "identity_n_op",
    deps = ARRAY_DEPS + [
        "//tensorflow/core:core_cpu_internal",
    ],
)

tf_kernel_library(
    name = "listdiff_op",
    prefix = "listdiff_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "one_hot_op",
    prefix = "one_hot_op",
    deps = ARRAY_DEPS + ["//tensorflow/core/util:overflow"],
)

tf_kernel_library(
    name = "pack_op",
    prefix = "pack_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "pad_op",
    prefix = "pad_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "quantize_and_dequantize_op",
    gpu_copts = tf_disable_ptxas_warning_flags(),
    prefix = "quantize_and_dequantize_op",
    deps = ARRAY_DEPS + [":cwise_op"],
)

tf_kernel_library(
    name = "reshape_op",
    features = ["-layering_check"],
    prefix = "reshape_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "reverse_op",
    prefix = "reverse_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "reverse_sequence_op",
    prefix = "reverse_sequence_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "shape_ops",
    features = ["-layering_check"],
    prefix = "shape_ops",
    deps = ARRAY_DEPS + ["//tensorflow/core/common_runtime:dma_helper"],
)

tf_kernel_library(
    name = "slice_op",
    prefix = "slice_op",
    deps = ARRAY_DEPS + [":strided_slice_op"],
)

tf_kernel_library(
    name = "snapshot_op",
    prefix = "snapshot_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "split_op",
    features = ["-layering_check"],
    gpu_srcs = ["gpu_device_array.h"],
    prefix = "split_op",
    deps = ARRAY_DEPS + [
        ":loose_headers",
        ":split_lib",
    ],
)

tf_kernel_library(
    name = "split_v_op",
    features = ["-layering_check"],
    gpu_srcs = ["gpu_device_array.h"],
    prefix = "split_v_op",
    deps = ARRAY_DEPS + [
        ":loose_headers",
        ":split_lib",
    ],
)

tf_kernel_library(
    name = "searchsorted_op",
    prefix = "searchsorted_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "inplace_ops",
    prefix = "inplace_ops",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "tile_ops",
    srcs = [
        "tile_functor_cpu.h",
        "tile_functor_cpu_bfloat16.cc",
        "tile_functor_cpu_bool.cc",
        "tile_functor_cpu_complex128.cc",
        "tile_functor_cpu_complex64.cc",
        "tile_functor_cpu_double.cc",
        "tile_functor_cpu_float.cc",
        "tile_functor_cpu_half.cc",
        "tile_functor_cpu_int16.cc",
        "tile_functor_cpu_int32.cc",
        "tile_functor_cpu_int64.cc",
        "tile_functor_cpu_int8.cc",
        "tile_functor_cpu_tstring.cc",
        "tile_functor_cpu_uint32.cc",
        "tile_functor_cpu_uint64.cc",
        "tile_functor_cpu_uint8.cc",
        "tile_functor_cpu_variant.cc",
    ],
    hdrs = ["tile_functor.h"],
    gpu_copts = tf_disable_ptxas_warning_flags(),
    gpu_srcs = [
        "tile_functor.h",
        "tile_functor_gpu.h",
        "tile_functor_gpu_bfloat16.cu.cc",
        "tile_functor_gpu_bool.cu.cc",
        "tile_functor_gpu_complex64.cu.cc",
        "tile_functor_gpu_complex128.cu.cc",
        "tile_functor_gpu_double.cu.cc",
        "tile_functor_gpu_float.cu.cc",
        "tile_functor_gpu_half.cu.cc",
        "tile_functor_gpu_int16.cu.cc",
        "tile_functor_gpu_int32.cu.cc",
        "tile_functor_gpu_int64.cu.cc",
    ],
    prefix = "tile_ops",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "transpose_op",
    srcs = [
        "transpose_op.cc",
    ],
    hdrs = ["transpose_op.h"],
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "unique_op",
    features = if_cuda(["-layering_check"]),
    prefix = "unique_op",
    deps = ARRAY_DEPS + [
        "@com_google_absl//absl/container:flat_hash_map",
    ] + if_cuda_or_rocm([
        ":gpu_prim_hdrs",
        ":gpu_prim_helpers",
    ]) + if_cuda([
        "//tensorflow/core/util:cuda_solvers",
    ]) + if_rocm([
        "//tensorflow/core/util:rocm_solvers",
    ]),
)

tf_kernel_library(
    name = "unpack_op",
    prefix = "unpack_op",
    deps = ARRAY_DEPS + [":split_lib"],
)

tf_kernel_library(
    name = "unravel_index_op",
    prefix = "unravel_index_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "where_op",
    srcs = ["where_op.cc"],
    hdrs = ["where_op.h"],
    features = ["-layering_check"],
    gpu_srcs = [
        "where_op.h",
        "where_op_gpu.cu.h",
        "where_op_gpu_impl_1.cu.cc",
        "where_op_gpu_impl_2.cu.cc",
        "where_op_gpu_impl_3.cu.cc",
        "where_op_gpu_impl_4.cu.cc",
        "where_op_gpu_impl_5.cu.cc",
        "where_op_gpu_impl_6.cu.cc",
        "where_op_gpu_impl_7.cu.cc",
        "where_op_gpu_impl_8.cu.cc",
    ],
    deps = if_cuda([
               "//tensorflow/core/util:cuda_solvers",
           ]) + if_rocm([
               "//tensorflow/core/util:rocm_solvers",
           ]) + [":gpu_prim_hdrs"] +
           ARRAY_DEPS,
)

cc_library(
    name = "composite_tensor_variant",
    srcs = ["composite_tensor_variant.cc"],
    hdrs = ["composite_tensor_variant.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "composite_tensor_variant_test",
    size = "small",
    srcs = ["composite_tensor_variant_test.cc"],
    deps = [
        ":composite_tensor_variant",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "composite_tensor_ops",
    srcs = ["composite_tensor_ops.cc"],
    deps = [
        ":composite_tensor_variant",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "ragged_ops",
    deps = [
        ":ragged_cross_op",
        ":ragged_fill_empty_rows_op",
        ":ragged_gather_op",
        ":ragged_range_op",
        ":ragged_tensor_from_variant_op",
        ":ragged_tensor_to_sparse_kernel",
        ":ragged_tensor_to_tensor_op",
        ":ragged_tensor_to_variant_op",
    ],
)

cc_library(
    name = "ragged_utils",
    hdrs = [
        "ragged_utils.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "ragged_gather_op",
    srcs = ["ragged_gather_op.cc"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

tf_cc_test(
    name = "ragged_gather_op_test",
    size = "small",
    srcs = ["ragged_gather_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ragged_gather_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "ragged_range_op",
    srcs = ["ragged_range_op.cc"],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

tf_cc_test(
    name = "ragged_range_op_test",
    srcs = ["ragged_range_op_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":ops_testutil",
        ":ragged_range_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "ragged_tensor_to_sparse_kernel",
    srcs = ["ragged_tensor_to_sparse_kernel.cc"],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

tf_cc_test(
    name = "ragged_tensor_to_tensor_op_test",
    size = "small",
    srcs = ["ragged_tensor_to_tensor_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ragged_tensor_to_tensor_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "ragged_tensor_to_tensor_op",
    srcs = ["ragged_tensor_to_tensor_op.cc"],
    deps = [
        ":broadcast_to_op",
        ":list_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util:ragged_to_dense_util",
    ],
)

tf_cc_test(
    name = "ragged_tensor_to_sparse_kernel_test",
    size = "small",
    srcs = ["ragged_tensor_to_sparse_kernel_test.cc"],
    deps = [
        ":ops_testutil",
        ":ragged_tensor_to_sparse_kernel",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "ragged_tensor_variant",
    srcs = ["ragged_tensor_variant.cc"],
    hdrs = ["ragged_tensor_variant.h"],
    features = ["-layering_check"],
    deps = [
        ":cwise_op",
        "//tensorflow/core:framework",
    ],
)

tf_kernel_library(
    name = "ragged_tensor_to_variant_op",
    srcs = ["ragged_tensor_to_variant_op.cc"],
    deps = [
        ":concat_lib",
        ":ragged_tensor_variant",
        ":ragged_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "ragged_tensor_from_variant_op",
    srcs = ["ragged_tensor_from_variant_op.cc"],
    deps = [
        ":ragged_tensor_variant",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "ragged_tensor_to_variant_op_test",
    size = "small",
    srcs = [
        "ragged_tensor_to_variant_op_test.cc",
        "ragged_tensor_to_variant_op_test.h",
    ],
    features = ["-layering_check"],
    deps = [
        ":ops_testutil",
        ":ragged_tensor_to_variant_op",
        ":ragged_tensor_variant",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:status_matchers",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "ragged_tensor_to_variant_op_large_data_test",
    timeout = "moderate",  # Set timeout to 300 sec.
    srcs = [
        "ragged_tensor_to_variant_op_large_data_test.cc",
        "ragged_tensor_to_variant_op_test.h",
    ],
    features = ["-layering_check"],
    tags = [
        "local",
        "manual",
        "notap",
    ],
    deps = [
        ":ops_testutil",
        ":ragged_tensor_to_variant_op",
        ":ragged_tensor_variant",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "ragged_tensor_from_variant_op_test",
    size = "small",
    srcs = ["ragged_tensor_from_variant_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ragged_tensor_from_variant_op",
        ":ragged_tensor_variant",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "ragged_cross_op",
    srcs = ["ragged_cross_op.cc"],
    features = ["-layering_check"],
    deps = [
        ":ragged_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "ragged_fill_empty_rows_op",
    prefix = "ragged_fill_empty_rows_op",
    deps = [
        "fill_empty_rows_functor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ] + [":gpu_prim_hdrs"] + if_cuda_or_rocm([
        ":gpu_prim_helpers",
    ]) + if_cuda([
        "//tensorflow/core/util:cuda_solvers",
    ]) + if_rocm([
        "//tensorflow/core/util:rocm_solvers",
    ]),
)

tf_cc_test(
    name = "ragged_fill_empty_rows_op_test",
    size = "small",
    srcs = ["ragged_fill_empty_rows_op_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":ops_testutil",
        ":ragged_fill_empty_rows_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "rnn_ops",
    deps = [
        "//tensorflow/core/kernels/rnn:gru_ops",
        "//tensorflow/core/kernels/rnn:lstm_ops",
    ],
)

tf_kernel_library(
    name = "cudnn_rnn_kernels",
    srcs = ["cudnn_rnn_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":cast_op",
        ":gpu_utils",
        ":numeric_options_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "batch_norm_op_test",
    size = "small",
    srcs = ["batch_norm_op_test.cc"],
    tags = [
        "no_oss",  # b/189866692
    ],
    deps = [
        ":batch_norm_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "batch_kernel_test_util",
    testonly = 1,
    srcs = ["batch_kernel_test_util.cc"],
    hdrs = ["batch_kernel_test_util.h"],
    features = ["-layering_check"],
    deps = [
        ":batch_kernels",
        ":ops_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "batch_kernels_test",
    size = "small",
    srcs = ["batch_kernels_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":batch_kernel_test_util",
        ":batch_kernels",
        ":cwise_op",
        ":function_ops",
        ":shape_ops",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels/batching_util:batch_scheduler_hdrs",
        "//tensorflow/core/kernels/batching_util:warmup",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/public:version",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:refcount",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_cc_test(
    name = "batch_kernels_auto_warmup_test",
    size = "small",
    srcs = ["batch_kernels_auto_warmup_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":batch_kernel_test_util",
        ":batch_kernels",
        ":function_ops",
        ":shape_ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels/batching_util:warmup",
        "@local_tsl//tsl/platform:blocking_counter",
    ],
)

tf_cc_test(
    name = "batch_kernels_env_test",
    size = "small",
    srcs = ["batch_kernels_env_test.cc"],
    deps = [
        ":batch_kernel_test_util",
        ":batch_kernels",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "ops_testutil_test",
    size = "small",
    srcs = ["ops_testutil_test.cc"],
    deps = [
        ":identity_op",
        ":ops_testutil",
        ":ops_util",
        ":variable_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "concat_op_test",
    size = "small",
    srcs = ["concat_op_test.cc"],
    deps = [
        ":concat_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/base:prefetch",
    ],
)

tf_cuda_cc_test(
    name = "bincount_op_test",
    size = "small",
    srcs = ["bincount_op_test.cc"],
    deps = [
        ":bincount_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "broadcast_to_op_test",
    size = "small",
    srcs = ["broadcast_to_op_test.cc"],
    deps = [
        ":broadcast_to_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "constant_op_test",
    size = "small",
    srcs = ["constant_op_test.cc"],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "deep_conv2d_test",
    size = "small",
    srcs = ["deep_conv2d_test.cc"],
    deps = [
        ":conv_ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_cc_test(
    name = "conv_ops_test",
    size = "medium",
    srcs = ["conv_ops_test.cc"],
    tags = [
        "no_cuda_asan",  # TODO(b/171342275): re-enable.
        "no_mac_arm64",
        "requires-gpu-sm70",
    ],
    deps = [
        ":conv_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels/image",
        "//tensorflow/core/platform:tensor_float_32_utils",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cuda_cc_test(
    name = "conv_ops_benchmark_test",
    size = "medium",
    srcs = ["conv_ops_benchmark_test.cc"],
    tags = [
        "nomac",  # b/132448918
    ],
    deps = [
        ":bias_op",
        ":conv_ops",
        ":fused_batch_norm_op",
        ":ops_testutil",
        ":ops_util",
        ":relu_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_cuda([
        "@local_xla//xla/stream_executor/cuda:cudnn_plugin",
    ]) + if_mkl([
        "//tensorflow/core/kernels/mkl:mkl_conv_op",
    ]),
)

tf_cuda_cc_test(
    name = "conv_grad_filter_ops_benchmark_test",
    size = "medium",
    srcs = ["conv_grad_filter_ops_benchmark_test.cc"],
    deps = [
        ":conv_ops",
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_cuda([
        "@local_xla//xla/stream_executor/cuda:cudnn_plugin",
    ]),
)

tf_cuda_cc_test(
    name = "conv_grad_input_ops_benchmark_test",
    size = "medium",
    srcs = ["conv_grad_input_ops_benchmark_test.cc"],
    deps = [
        ":conv_ops",
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_cuda([
        "@local_xla//xla/stream_executor/cuda:cudnn_plugin",
    ]),
)

tf_cuda_cc_test(
    name = "depthwise_conv_ops_test",
    size = "small",
    srcs = ["depthwise_conv_ops_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171342266): re-enable.
        "no_gpu",  # TODO(b/194100358): re-enable after flakiness resolved.
    ],
    deps = [
        ":conv_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels/image",
    ],
)

tf_cc_test(
    name = "example_parsing_ops_test",
    size = "medium",
    srcs = ["example_parsing_ops_test.cc"],
    shard_count = 4,
    tags = ["optonly"],
    deps = [
        ":example_parsing_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/base",
    ],
)

tf_cuda_cc_test(
    name = "fake_quant_ops_test",
    size = "small",
    srcs = ["fake_quant_ops_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":fake_quant_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "fused_batch_norm_op_test",
    size = "small",
    srcs = ["fused_batch_norm_op_test.cc"],
    deps = [
        ":fused_batch_norm_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_cuda([
        "@local_xla//xla/stream_executor/cuda:cudnn_plugin",
    ]),
)

tf_cuda_cc_test(
    name = "fused_batch_norm_ex_op_test",
    size = "small",
    srcs = ["fused_batch_norm_ex_op_test.cc"],
    features = if_cuda(["-layering_check"]),
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":cwise_op",
        ":fused_batch_norm_op",
        ":ops_testutil",
        ":ops_util",
        ":relu_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ] + if_cuda([
        "@local_xla//xla/stream_executor/cuda:cudnn_plugin",
    ]),
)

tf_cc_test(
    name = "in_topk_op_test",
    size = "small",
    srcs = ["in_topk_op_test.cc"],
    deps = [
        ":in_topk_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_cuda([
        "@local_xla//xla/stream_executor/cuda:cudnn_plugin",
    ]),
)

tf_kernel_library(
    name = "gather_functor",
    features = ["-layering_check"],
    prefix = "gather_functor",
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:bounds_check",
        "@com_google_absl//absl/base:prefetch",
        "@eigen_archive//:eigen3",
    ],
)

# Unlike gather_functor library, this does not include the CUDA code and deps.
cc_library(
    name = "gather_functor_hdr",
    hdrs = [
        "gather_functor.h",
        "gather_functor_batched.h",
    ],
)

tf_kernel_library(
    name = "dense_update_functor",
    srcs = ["dense_update_functor.cc"],
    hdrs = ["dense_update_functor.h"],
    gpu_srcs = [
        "dense_update_functor.h",
        "dense_update_functor_gpu.cu.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 0,
)

tf_cuda_cc_test(
    name = "gather_op_test",
    size = "medium",
    srcs = ["gather_op_test.cc"],
    deps = [
        ":gather_op",
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "gather_nd_op_test",
    size = "small",
    srcs = ["gather_nd_op_test.cc"],
    deps = [
        ":gather_nd_op",
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "guarantee_const_op_test",
    size = "small",
    srcs = ["guarantee_const_op_test.cc"],
    deps = [
        ":guarantee_const_op",
        ":ops_testutil",
        ":ops_util",
        ":variable_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "identity_op_test",
    size = "small",
    srcs = ["identity_op_test.cc"],
    deps = [
        ":identity_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "identity_n_op_test",
    size = "small",
    srcs = ["identity_n_op_test.cc"],
    deps = [
        ":identity_n_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "debug_ops_test",
    size = "small",
    srcs = ["debug_ops_test.cc"],
    deps = [
        ":debug_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:debug_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/debug:debug_io_utils",
        "//tensorflow/core/debug:debug_node_key",
    ],
)

tf_cuda_cc_test(
    name = "quantize_and_dequantize_op_test",
    size = "small",
    srcs = ["quantize_and_dequantize_op_test.cc"],
    tags = [
        "no_windows",  # test uses rand_r which does not exist on Windows
    ],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantize_and_dequantize_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:status_matchers",
    ],
)

tf_cc_test(
    name = "dequantize_op_test",
    size = "small",
    srcs = ["dequantize_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "one_hot_op_test",
    size = "small",
    srcs = ["one_hot_op_test.cc"],
    deps = [
        ":one_hot_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_cuda([
        "@local_xla//xla/stream_executor/cuda:cudnn_plugin",
    ]),
)

tf_cc_test(
    name = "reverse_op_test",
    size = "small",
    srcs = ["reverse_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":reverse_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "scatter_functor",
    prefix = "scatter_functor",
    visibility = [":friends"],
    deps = [
        ":dense_update_functor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/util:determinism_for_kernels",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "shape_ops_test",
    size = "small",
    srcs = ["shape_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":shape_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "slice_op_test",
    size = "small",
    srcs = ["slice_op_test.cc"],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":slice_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "strided_slice_op_test",
    size = "small",
    srcs = ["strided_slice_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":slice_op",
        ":strided_slice_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "unique_op_test",
    size = "small",
    srcs = ["unique_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":unique_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "transpose_functor",
    srcs = ["transpose_functor_cpu.cc"],
    hdrs = ["transpose_functor.h"],
    gpu_srcs = [
        "transpose_functor_gpu.cu.cc",
        "transpose_functor.h",
    ],
    visibility = [":friends"],
    deps = [
        ":conv_2d",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "transpose_util_test",
    size = "small",
    srcs = ["transpose_util_test.cc"],
    deps = [
        ":transpose_functor",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_testutil",
    ],
)

tf_kernel_library(
    name = "candidate_sampler_ops",
    prefix = "candidate_sampler_ops",
    deps = [
        ":range_sampler",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "range_sampler",
    srcs = ["range_sampler.cc"],
    hdrs = ["range_sampler.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "range_sampler_test",
    size = "small",
    srcs = ["range_sampler_test.cc"],
    deps = [
        ":range_sampler",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "control_flow_ops",
    prefix = "control_flow_ops",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "while_op_test",
    srcs = ["while_op_test.cc"],
    features = ["-layering_check"],
    tags = [
        "no_windows",
    ],  # TODO(b/208697533): Re-enable after fixing.
    deps = [
        ":control_flow_ops",
        "//tensorflow/c/experimental/stream_executor",
        "//tensorflow/c/experimental/stream_executor:stream_executor_test_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/common_runtime/pluggable_device:pluggable_device_runtime_impl",
        "//tensorflow/core/framework:function_testlib",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:functional_ops",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_kernel_library(
    name = "ctc_ops",
    prefix = "ctc",
    deps = [
        ":loose_headers",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/util/ctc:ctc_beam_search_lib",
        "//tensorflow/core/util/ctc:ctc_loss_calculator_lib",
    ] + if_cuda_or_rocm([
        ":gpu_utils",
        ":conv_ops_gpu_hdrs",
    ]) + if_cuda([
        "@local_config_cuda//cuda:cudnn_header",
    ]) + if_rocm([
        "//tensorflow/core/platform:stream_executor",
    ]),
)

tf_cc_test(
    name = "control_flow_ops_test",
    size = "small",
    srcs = ["control_flow_ops_test.cc"],
    deps = [
        ":control_flow_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:type_inference",
    ],
)

cc_library(
    name = "data_flow",
    deps = [
        ":barrier_ops",
        ":conditional_accumulator_base_op",
        ":conditional_accumulator_op",
        ":dynamic_partition_op",
        ":dynamic_stitch_op",
        ":fifo_queue_op",
        ":lookup_table_init_op",
        ":lookup_table_op",
        ":map_stage_op",
        ":padding_fifo_queue_op",
        ":priority_queue_op",
        ":queue_ops",
        ":random_shuffle_queue_op",
        ":record_input_op",
        ":session_ops",
        ":sparse_conditional_accumulator_op",
        ":stack_ops",
        ":stage_op",
        ":tensor_array_ops",
    ],
)

cc_library(
    name = "lookup",
    deps = [
        ":lookup_table_init_op",
        ":lookup_table_op",
    ],
)

cc_header_only_library(
    name = "lookup_headers_lib",
    deps = [":lookup"],
)

DATA_FLOW_DEPS = [
    ":concat_lib",
    ":conditional_accumulator",
    ":conditional_accumulator_base",
    ":fifo_queue",
    ":initializable_lookup_table",
    ":lookup_util",
    ":padding_fifo_queue",
    ":priority_queue",
    ":queue_base",
    ":queue_op",
    ":sparse_conditional_accumulator",
    ":split_lib",
    ":tensor_array",
    ":typed_conditional_accumulator_base",
    ":typed_queue",
    "@eigen_archive//:eigen3",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core/framework:bounds_check",
]

tf_kernel_library(
    name = "conditional_accumulator_base_op",
    prefix = "conditional_accumulator_base_op",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "conditional_accumulator_op",
    prefix = "conditional_accumulator_op",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "barrier_ops",
    prefix = "barrier_ops",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "fifo_queue_op",
    prefix = "fifo_queue_op",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "padding_fifo_queue_op",
    prefix = "padding_fifo_queue_op",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "priority_queue_op",
    prefix = "priority_queue_op",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "queue_ops",
    prefix = "queue_ops",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "random_shuffle_queue_op",
    prefix = "random_shuffle_queue_op",
    deps = DATA_FLOW_DEPS + [
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "scoped_allocator_ops",
    prefix = "scoped_allocator_ops",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cuda_cc_test(
    name = "scoped_allocator_ops_test",
    srcs = ["scoped_allocator_ops_test.cc"],
    deps = [
        ":cwise_op",
        ":dense_update_ops",
        ":ops_testutil",
        ":ops_util",
        ":scoped_allocator_ops",
        ":variable_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "session_ops",
    prefix = "session_ops",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "sparse_conditional_accumulator_op",
    prefix = "sparse_conditional_accumulator_op",
    deps = DATA_FLOW_DEPS,
)

cc_library(
    name = "stack",
    srcs = ["stack.cc"],
    hdrs = ["stack.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "stack_ops",
    prefix = "stack_ops",
    deps = DATA_FLOW_DEPS + [":stack"],
)

tf_kernel_library(
    name = "tensor_array_ops",
    prefix = "tensor_array_ops",
    deps = DATA_FLOW_DEPS,
)

DYNAMIC_DEPS = [
    "//tensorflow/core/framework:bounds_check",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
]

tf_kernel_library(
    name = "dynamic_partition_op",
    features = if_cuda(["-layering_check"]),
    prefix = "dynamic_partition_op",
    deps = DYNAMIC_DEPS + [
        ":fill_functor",
        ":gather_functor",
        ":gpu_prim_hdrs",
        "//tensorflow/core:framework_internal",
    ],
)

tf_kernel_library(
    name = "dynamic_stitch_op",
    gpu_srcs = [
        "gpu_device_array.h",
        "gpu_device_array_gpu.h",
    ],
    prefix = "dynamic_stitch_op",
    deps = DYNAMIC_DEPS + [":loose_headers"],
)

cc_library(
    name = "tensor_cord",
    srcs = ["tensor_cord.cc"],
    hdrs = ["tensor_cord.h"],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core:framework",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "tensor_cord_test",
    srcs = ["tensor_cord_test.cc"],
    deps = [
        ":tensor_cord",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:cord",
    ],
)

LOOKUP_DEPS = [
    ":initializable_lookup_table",
    ":lookup_util",
    "@com_google_absl//absl/container:flat_hash_map",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core/framework:bounds_check",
]

tf_kernel_library(
    name = "lookup_table_init_op",
    prefix = "lookup_table_init_op",
    deps = LOOKUP_DEPS,
)

tf_kernel_library(
    name = "lookup_table_op",
    prefix = "lookup_table_op",
    deps = LOOKUP_DEPS,
)

cc_library(
    name = "checkpoint_ops",
    deps = [
        ":generate_vocab_remapping_op",
        ":load_and_remap_matrix_op",
    ],
)

tf_kernel_library(
    name = "generate_vocab_remapping_op",
    srcs = ["generate_vocab_remapping_op.cc"],
    deps = [
        ":lookup_table_init_op",
        ":lookup_table_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "load_and_remap_matrix_op",
    srcs = ["load_and_remap_matrix_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/util/tensor_bundle",
        "@eigen_archive//:eigen3",
    ],
)

tf_cuda_cc_tests(
    name = "dynamic_op_test",
    size = "small",
    srcs = [
        "dynamic_partition_op_test.cc",
        "dynamic_stitch_op_test.cc",
    ],
    deps = [
        ":data_flow",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "fifo_queue",
    srcs = ["fifo_queue.cc"],
    hdrs = ["fifo_queue.h"],
    visibility = [":friends"],
    deps = [
        ":queue_base",
        ":queue_op",
        ":typed_queue",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "padding_fifo_queue",
    srcs = ["padding_fifo_queue.cc"],
    hdrs = ["padding_fifo_queue.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":fifo_queue",
        ":queue_base",
        ":typed_queue",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "conditional_accumulator_base",
    srcs = ["conditional_accumulator_base.cc"],
    hdrs = [
        "conditional_accumulator_base.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "typed_conditional_accumulator_base",
    hdrs = ["typed_conditional_accumulator_base.h"],
    deps = [
        ":conditional_accumulator_base",
    ],
)

cc_library(
    name = "conditional_accumulator",
    hdrs = [
        "conditional_accumulator.h",
        "conditional_accumulator_base_op.h",
    ],
    deps = [
        ":conditional_accumulator_base",
        ":fill_functor",
        ":typed_conditional_accumulator_base",
    ],
)

cc_library(
    name = "sparse_conditional_accumulator",
    hdrs = ["sparse_conditional_accumulator.h"],
    deps = [
        ":typed_conditional_accumulator_base",
    ],
)

tf_kernel_library(
    name = "tensor_array",
    srcs = ["tensor_array.cc"],
    hdrs = ["tensor_array.h"],
    features = ["-layering_check"],
    visibility = ["//visibility:private"],
    deps = [
        ":aggregate_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "resource_variable_ops",
    srcs = ["resource_variable_ops.cc"],
    hdrs = ["resource_variable_ops.h"],
    features = ["-layering_check"],
    deps = [
        ":dense_update_functor",
        ":gather_functor",
        ":gather_nd_op",
        ":resource_variable_util",
        ":scatter_functor",
        ":training_op_helpers",
        ":variable_ops",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/util:determinism_for_kernels",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "resource_variable_util",
    srcs = ["resource_variable_util.cc"],
    hdrs = ["resource_variable_util.h"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tensor_list",
    srcs = ["tensor_list.cc"],
    hdrs = ["tensor_list.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:tensor_shape_proto_cc",
        "//tensorflow/core/lib/core:refcount",
    ],
)

cc_library(
    name = "tensor_list_util",
    srcs = ["tensor_list_util.cc"],
    hdrs = ["tensor_list_util.h"],
    deps = [
        ":tensor_list",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "list_kernels",
    srcs = ["list_kernels.cc"],
    hdrs = ["list_kernels.h"],
    features = ["-layering_check"],
    gpu_srcs = [
        "list_kernels.cu.cc",
        "list_kernels.h",
    ],
    deps = [
        ":concat_lib",
        ":fill_functor",
        ":tensor_list",
        ":tensor_list_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/stream_executor:stream",
    ],
)

cc_library(
    name = "tensor_map",
    srcs = ["tensor_map.cc"],
    hdrs = ["tensor_map.h"],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:tensor_shape_proto_cc",
        "//tensorflow/core/lib/core:refcount",
    ],
)

tf_kernel_library(
    name = "map_kernels",
    srcs = ["map_kernels.cc"],
    hdrs = ["map_kernels.h"],
    deps = [
        ":concat_lib",
        ":fill_functor",
        ":tensor_map",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_tests(
    name = "tensor_map_test",
    size = "small",
    srcs = [
        "tensor_map_test.cc",
    ],
    deps = [
        ":tensor_map",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_testutil",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "fact_op",
    prefix = "fact_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "function_ops",
    features = ["-layering_check"],
    prefix = "function_ops",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

tf_kernel_library(
    name = "functional_ops",
    prefix = "functional_ops",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "partitioned_function_ops",
    prefix = "partitioned_function_ops",
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/grappler/utils:functions",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_tests(
    name = "eigen_test",
    size = "small",
    srcs = [
        "eigen_activations_test.cc",
        "eigen_attention_test.cc",
        "eigen_backward_cuboid_convolutions_test.cc",
        "eigen_backward_spatial_convolutions_test.cc",
        "eigen_pooling_test.cc",
    ],
    deps = [
        ":eigen_helpers",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

# Conditional test target generation is not supported by the "tf_cc_tests" macro
# (can't add 'select' to the srcs field, type 'select' is not iterable).
tf_cc_test(
    name = "eigen_mkldnn_contraction_kernel_test",
    size = "small",
    srcs = select({
        "//tensorflow:android_x86": [],
        "//tensorflow:arm_any": [],
        "//tensorflow:fuchsia_x86_64": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_ppc64le": [],
        "//tensorflow:linux_s390x": [],
        ":no_mkldnn_contraction_kernel": [],
        "//conditions:default": ["eigen_mkldnn_contraction_kernel_test.cc"],
    }),
    features = ["-layering_check"],
    tags = ["mkldnn_contraction_kernel"],
    deps = [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel",
    ],
)

cc_library(
    name = "eigen_benchmark",
    testonly = 1,
    hdrs = [
        "eigen_benchmark.h",
        ":eigen_helpers",
    ],
    deps = [
        "//tensorflow/core:framework",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "eigen_benchmark_cpu_test",
    srcs = ["eigen_benchmark_cpu_test.cc"],
    deps = [
        ":eigen_benchmark",
        ":eigen_helpers",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_tests(
    name = "basic_ops_benchmark_test",
    size = "small",
    srcs = [
        "basic_ops_benchmark_test.cc",
    ],
    deps = [
        ":math",
        ":ops_util",
        ":state",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "io",
    deps = [
        ":fixed_length_record_reader_op",
        ":identity_reader_op",
        ":matching_files_op",
        ":reader_ops",
        ":restore_op",
        ":save_op",
        ":save_restore_v2_ops",
        ":text_line_reader_op",
        ":tf_record_reader_op",
        ":whole_file_read_ops",
    ] + select({
        "//tensorflow:fuchsia": [],
        "//conditions:default": [":lmdb_reader_op"],
    }),
)

IO_DEPS = [
    ":ops_util",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core:reader_base",
    "//tensorflow/core/util/tensor_bundle",
]

tf_kernel_library(
    name = "fixed_length_record_reader_op",
    prefix = "fixed_length_record_reader_op",
    deps = IO_DEPS,
)

tf_kernel_library(
    name = "identity_reader_op",
    prefix = "identity_reader_op",
    deps = IO_DEPS + ["@com_google_absl//absl/strings"],
)

tf_kernel_library(
    name = "lmdb_reader_op",
    prefix = "lmdb_reader_op",
    deps = IO_DEPS,
)

tf_kernel_library(
    name = "matching_files_op",
    prefix = "matching_files_op",
    deps = IO_DEPS,
)

tf_kernel_library(
    name = "reader_ops",
    prefix = "reader_ops",
    deps = IO_DEPS,
)

SAVE_RESTORE_DEPS = [
    ":checkpoint_callback_manager",
    ":save_restore_tensor",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core/framework:bounds_check",
    "//tensorflow/core/util/tensor_bundle",
    "//tensorflow/core/util/tensor_bundle:naming",
]

tf_kernel_library(
    name = "restore_op",
    prefix = "restore_op",
    deps = SAVE_RESTORE_DEPS,
)

tf_kernel_library(
    name = "save_op",
    prefix = "save_op",
    deps = SAVE_RESTORE_DEPS,
)

tf_kernel_library(
    name = "checkpoint_callback_manager",
    srcs = [
        "checkpoint_callback_manager.cc",
    ],
    hdrs = [
        "checkpoint_callback_manager.h",
    ],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:regexp",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_tests(
    name = "checkpoint_callback_manager_test",
    size = "small",
    srcs = ["checkpoint_callback_manager_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":checkpoint_callback_manager",
        ":io",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "save_restore_v2_ops",
    prefix = "save_restore_v2_ops",
    deps = SAVE_RESTORE_DEPS,
)

tf_kernel_library(
    name = "text_line_reader_op",
    prefix = "text_line_reader_op",
    deps = IO_DEPS + ["@com_google_absl//absl/status"],
)

tf_kernel_library(
    name = "tf_record_reader_op",
    prefix = "tf_record_reader_op",
    deps = IO_DEPS + ["@com_google_absl//absl/status"],
)

tf_kernel_library(
    name = "whole_file_read_ops",
    prefix = "whole_file_read_ops",
    deps = IO_DEPS + ["@com_google_absl//absl/strings"],
)

tf_cc_tests(
    name = "bonus2_tests",
    size = "small",
    srcs = [
        "merge_v2_checkpoints_op_test.cc",
        "restore_op_test.cc",
        "restore_v2_op_test.cc",
        "save_op_test.cc",
        "save_v2_op_test.cc",
    ],
    deps = [
        ":io",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

cc_library(
    name = "logging",
    deps = [
        ":logging_ops",
        ":summary_audio_op",
        ":summary_image_op",
        ":summary_tensor_op",
    ],
)

LOGGING_DEPS = [
    "@com_google_absl//absl/strings",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:protos_all_cc",
    # TODO(b/162630222): remove this dependency.
    "//tensorflow/c/kernels:histogram_summary_op",
    "//tensorflow/c/kernels:merge_summary_op",
    "//tensorflow/c/kernels:summary_op",
    "//tensorflow/core/util:determinism_for_kernels",
]

tf_kernel_library(
    name = "logging_ops",
    prefix = "logging_ops",
    deps = LOGGING_DEPS,
)

tf_kernel_library(
    name = "summary_audio_op",
    prefix = "summary_audio_op",
    deps = LOGGING_DEPS,
)

tf_kernel_library(
    name = "summary_image_op",
    prefix = "summary_image_op",
    deps = LOGGING_DEPS + ["//tensorflow/core/lib/png:png_io"],
)

# TODO(b/162630222): remove this target
cc_library(
    name = "summary_op",
    deps = [
        "//tensorflow/c/kernels:histogram_summary_op",
        "//tensorflow/c/kernels:merge_summary_op",
        "//tensorflow/c/kernels:summary_op",
    ],
)

tf_kernel_library(
    name = "summary_tensor_op",
    prefix = "summary_tensor_op",
    deps = LOGGING_DEPS,
)

tf_cc_tests(
    name = "bonus3_tests",
    size = "small",
    srcs = [
        "logging_ops_test.cc",
        "summary_audio_op_test.cc",
        "summary_image_op_test.cc",
        "summary_op_test.cc",
        "summary_tensor_op_test.cc",
    ],
    deps = [
        ":logging",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:status_matchers",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/util:determinism_test_util",
    ],
)

cc_library(
    name = "manip",
    deps = [
        ":roll_op",
    ],
)

tf_kernel_library(
    name = "roll_op",
    srcs = ["roll_op.cc"],
    hdrs = ["roll_op.h"],
    gpu_srcs = [
        "roll_op_gpu.cu.cc",
        "roll_op.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "roll_op_test",
    size = "small",
    srcs = ["roll_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":roll_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "resource_ops_test",
    size = "small",
    srcs = ["resource_ops_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":dense_update_functor",
        ":ops_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:types_proto_cc",
    ],
)

tf_cc_test(
    name = "lookup_ops_test",
    size = "small",
    srcs = ["lookup_ops_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":lookup_table_op",
        ":ops_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

MATH_DEPS = [
    ":fill_functor",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:math_grad",
    "//tensorflow/core/framework:bounds_check",
    "//tensorflow/core/framework:op_requires",
    "@eigen_archive//:eigen3",
]

tf_kernel_library(
    name = "sparse_matmul_op",
    prefix = "sparse_matmul_op",
    deps = MATH_DEPS + ["@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel"],
)

cc_library(
    name = "math",
    deps = [
        ":aggregate_ops",
        ":argmax_op",
        ":betainc_op",
        ":bincount_op",
        ":bucketize_op",
        ":cast_op",
        ":check_numerics_op",
        ":cross_op",
        ":cwise_op",
        ":fft_ops",
        ":histogram_op",
        ":matmul_op",
        ":nextafter_op",
        ":population_count_op",
        ":reduction_ops",
        ":scan_ops",
        ":segment_reduction_ops",
        ":sequence_ops",
        ":sparse_matmul_op",
        "//tensorflow/core/kernels/special_math:special_math_op",
    ],
)

tf_kernel_library(
    name = "aggregate_ops",
    prefix = "aggregate_ops",
    deps = MATH_DEPS + [":variant_ops_util"],
)

tf_kernel_library(
    name = "variant_ops_util",
    srcs = ["variant_ops_util.cc"],
    hdrs = ["variant_ops_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "argmax_op",
    prefix = "argmax_op",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "batch_matmul_op",
    deps = [":matmul_op"],
)

tf_kernel_library(
    name = "matmul_op",
    features = ["-layering_check"],
    prefix = "matmul_op",
    textual_hdrs = ["matmul_op_impl.h"],
    deps = MATH_DEPS + [
        ":fused_eigen_output_kernels",
        ":loose_headers",
        "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel",
    ] + mkl_deps() + if_cuda([
        "@local_xla//xla/stream_executor/cuda:cublas_plugin",
    ]) + if_rocm_hipblaslt([
        "@local_xla//xla/stream_executor/rocm:hipblas_lt_header",
    ]) + if_cuda_or_rocm([
        ":conv_ops_gpu_hdrs",
        ":gpu_utils",
        ":conv_ops",
        ":matmul_util",
        "@local_xla//xla/stream_executor/gpu:redzone_allocator",
        "@local_xla//xla/stream_executor/integrations:tf_allocator_adapter",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        ":numeric_options_utils",
        "//tensorflow/core/protobuf:autotuning_proto_cc",
        "//tensorflow/core/util/autotune_maps:conv_parameters",
        "//tensorflow/core/util/proto:proto_utils",
    ]),
)

cc_library(
    name = "matmul_util",
    srcs = ["matmul_util.cc"],
    hdrs = ["matmul_util.h"],
    features = ["-layering_check"],
    local_defines = if_cuda(["GOOGLE_CUDA=1"]) + if_rocm(["TENSORFLOW_USE_ROCM=1"]),
    deps = if_cuda_or_rocm([
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/stream_executor/cuda:cublas_plugin",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:tensor_float_32_hdr_lib",
        "//tensorflow/core/util:env_var",
    ]) + if_cuda([
        "@local_xla//xla/stream_executor/cuda:cublas_lt_header",
    ]) + if_rocm([
        "@local_xla//xla/stream_executor/rocm:hipblas_lt_header",
    ]) + if_static(["//tensorflow/core/platform:tensor_float_32_utils"]),
)

tf_kernel_library(
    name = "betainc_op",
    prefix = "betainc_op",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "bucketize_op",
    features = if_cuda(["-layering_check"]),
    gpu_srcs = ["gpu_device_array.h"],
    prefix = "bucketize_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "cast_op",
    copts = if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    features = ["-layering_check"],
    # *.cu.cc sources are compiled with gpu_copts instead of copts.
    gpu_copts = if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    prefix = "cast_op",
    deps = MATH_DEPS + [
        "//tensorflow/core/kernels/mlir_generated:cast_op",
    ],
)

tf_kernel_library(
    name = "check_numerics_op",
    features = ["-layering_check"],
    prefix = "check_numerics_op",
    deps = MATH_DEPS + ["//tensorflow/core:framework_internal"],
)

tf_kernel_library(
    name = "cross_op",
    prefix = "cross_op",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "cwise_op",
    copts = if_mlir_generated_experimental_kernels_enabled([
        "-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED",
    ]) + if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    # *.cu.cc sources are compiled with gpu_copts instead of copts.
    gpu_copts = if_mlir_generated_experimental_kernels_enabled([
        "-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED",
    ]) + if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    prefix = "cwise_op",
    deps = MATH_DEPS + [
        "//tensorflow/core/kernels/mlir_generated:cwise_op",
        "@com_google_absl//absl/base:prefetch",
    ],
)

tf_kernel_library(
    name = "nextafter_op",
    copts = if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    features = if_cuda(["-layering_check"]),
    # *.cu.cc sources are compiled with gpu_copts instead of copts.
    gpu_copts = if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    prefix = "nextafter_op",
    deps = MATH_DEPS + [
        ":cwise_op",
        "//tensorflow/core/kernels/mlir_generated:nextafter_op",
    ],
)

tf_kernel_library(
    name = "population_count_op",
    prefix = "population_count_op",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "fft_ops",
    features = ["-layering_check"],
    prefix = "fft_ops",
    deps = MATH_DEPS + if_cuda([
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/core/platform:stream_executor",
        "@local_xla//xla/stream_executor/cuda:cufft_plugin",
    ]) + ["@ducc//:fft_wrapper"],
)

tf_kernel_library(
    name = "reduction_ops",
    features = if_cuda(["-layering_check"]),
    gpu_srcs = ["reduction_gpu_kernels.cu.h"],
    prefix = "reduction_ops",
    deps = MATH_DEPS + [
        ":gpu_prim_hdrs",
        ":transpose_functor",
    ],
)

tf_kernel_library(
    name = "segment_reduction_ops",
    features = ["-layering_check"],
    prefix = "segment_reduction_ops",
    deps = MATH_DEPS + [
        "//tensorflow/core/util:determinism_for_kernels",
    ] + if_cuda_or_rocm([
        ":gpu_prim_helpers",
    ]) + if_cuda([
        "//tensorflow/core/util:cuda_solvers",
    ]) + if_rocm([
        "//tensorflow/core/util:rocm_solvers",
    ]),
)

tf_kernel_library(
    name = "scan_ops",
    srcs = ["scan_ops.cc"],
    hdrs = ["scan_ops.h"],
    features = if_cuda(["-layering_check"]),
    gpu_srcs = [
        "scan_ops.h",
        "scan_ops_gpu.h",
        "scan_ops_gpu_bfloat16.cu.cc",
        "scan_ops_gpu_double.cu.cc",
        "scan_ops_gpu_float.cu.cc",
        "scan_ops_gpu_half.cu.cc",
        "scan_ops_gpu_int.cu.cc",
    ],
    deps = MATH_DEPS + [":gpu_prim_hdrs"],
)

tf_kernel_library(
    name = "sequence_ops",
    prefix = "sequence_ops",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "unary_ops_composition",
    prefix = "unary_ops_composition",
    deps = MATH_DEPS + [
        ":cwise_op",
        ":relu_op",
    ],
)

tf_cc_test(
    name = "sequence_ops_test",
    size = "small",
    srcs = ["sequence_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":sequence_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "cast_op_test",
    size = "small",
    srcs = ["cast_op_test.cc"],
    deps = [
        ":cast_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "cross_op_test",
    size = "small",
    srcs = ["cross_op_test.cc"],
    deps = [
        ":cross_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_tests(
    name = "sparse_tests",
    size = "small",
    srcs = [
        "sparse_add_op_test.cc",
        "sparse_dense_binary_op_shared_test.cc",
        "sparse_reduce_sum_op_test.cc",
    ],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":sparse_add_op",
        ":sparse_dense_binary_op_shared",
        ":sparse_reduce_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "cwise_ops_test",
    size = "small",
    srcs = ["cwise_ops_test.cc"],
    deps = [
        ":cwise_op",
        ":nn",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:bounds_check",
    ],
)

tf_cuda_cc_test(
    name = "unary_ops_composition_test",
    size = "small",
    srcs = ["unary_ops_composition_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":unary_ops_composition",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "matmul_op_test",
    srcs = ["matmul_op_test.cc"],
    deps = [
        ":matmul_op",
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_cuda_cc_test(
    name = "scan_ops_test",
    size = "small",
    srcs = ["scan_ops_test.cc"],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    deps = [
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        ":scan_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "reduction_ops_test",
    size = "small",
    srcs = ["reduction_ops_test.cc"],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    deps = [
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        ":reduction_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "segment_reduction_ops_test",
    size = "small",
    srcs = ["segment_reduction_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":segment_reduction_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "immutable_constant_op_test",
    srcs = ["immutable_constant_op_test.cc"],
    deps = [
        ":array",
        ":immutable_constant_op",
        ":matmul_op",
        ":ops_testutil",
        ":ops_util",
        ":random_shuffle_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "sparse_matmul_op_test",
    size = "small",
    srcs = ["sparse_matmul_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":sparse_matmul_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "split_op_test",
    size = "small",
    srcs = ["split_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":split_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "split_v_op_test",
    size = "small",
    srcs = ["split_v_op_test.cc"],
    tags = [
        "no_windows",  # split_v_op uses lrand48 which does not exist on Windows
    ],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":split_v_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "diag_op_test",
    size = "small",
    srcs = ["diag_op_test.cc"],
    deps = [
        ":diag_op",
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# conv_grad_ops currently has to be built with conv_ops*.
# TODO(josh11b, zhengxq): put these a separate libraries in ":nn" below once
# conv_ops_gpu.h has be separated into its own library.
tf_kernel_library(
    name = "conv_ops",
    srcs = [
        "conv_grad_filter_ops.cc",
        "conv_grad_filter_ops_3d.cc",
        "conv_grad_filter_ops_launcher.cc",
        "conv_grad_input_ops.cc",
        "conv_grad_input_ops_3d.cc",
        "conv_grad_input_ops_bfloat16.cc",
        "conv_grad_input_ops_double.cc",
        "conv_grad_input_ops_float.cc",
        "conv_grad_input_ops_half.cc",
        "conv_grad_input_ops_int32.cc",
        "deep_conv2d.cc",
    ],
    hdrs = [
        "conv_grad_input_ops.h",
        "conv_grad_ops.h",
        "deep_conv2d.h",
        "fill_functor.h",
        "gemm_functors.h",
        "winograd_transform.h",
    ],
    defines = [
        "EIGEN_NEON_GEBP_NR=4",
    ],
    features = ["-layering_check"],
    prefix = "conv_ops",
    textual_hdrs = [
        "autotune_conv_impl.h",
        "conv_ops_fused_impl.h",
        "conv_ops_impl.h",
    ],
    deps = [
        ":cast_op",
        ":conv_2d",
        ":conv_3d",
        ":conv_grad_shape_utils",
        ":cwise_lib_hdrs",
        ":fill_functor",
        ":fused_eigen_output_kernels",
        ":loose_headers",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "//tensorflow/core/protobuf:autotuning_proto_cc",
        "//tensorflow/core/util:image_resizer_state",
        "//tensorflow/core/util/proto:proto_utils",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel",
    ] + if_cuda([
        ":numeric_options_utils",
        "@local_xla//xla/stream_executor/cuda:cublas_plugin",
        "@local_xla//xla/stream_executor/cuda:cudnn_plugin",
        "@local_xla//xla/stream_executor/integrations:tf_allocator_adapter",
        "//tensorflow/core/platform:stream_executor",
    ]) + if_cuda_or_rocm([
        ":gpu_utils",
        "@local_xla//xla/stream_executor/gpu:redzone_allocator",
        "//tensorflow/core/util/autotune_maps:conv_parameters",
        "//tensorflow/core/util/autotune_maps:conv_autotune_maps",
    ]),
)

cc_library(
    name = "conv_grad_shape_utils",
    srcs = [
        "conv_grad_shape_utils.cc",
    ],
    hdrs = [
        "conv_grad_shape_utils.h",
    ],
    deps = [
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
    ],
)

tf_kernel_library(
    name = "depthwise_conv_op",
    srcs = ["depthwise_conv_op.cc"],
    hdrs = ["depthwise_conv_op.h"],
    features = ["-layering_check"],
    gpu_copts = if_not_windows([
        "-Wno-pass-failed",  # clang misses #pragma loop optimizations
    ]),
    gpu_srcs = [
        "depthwise_conv_op.h",
        "depthwise_conv_op_gpu.h",
        "depthwise_conv_op_gpu_bfloat16.cu.cc",
        "depthwise_conv_op_gpu_double.cu.cc",
        "depthwise_conv_op_gpu_float.cu.cc",
        "depthwise_conv_op_gpu_half.cu.cc",
    ],
    deps = [
        ":conv_ops",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/util:determinism_for_kernels",
    ] + if_cuda([
        "@local_config_cuda//cuda:cub_headers",
        "@local_config_cuda//cuda:cudnn_header",
    ]) + if_rocm([
        "@local_config_rocm//rocm:rocprim",
    ]) + if_cuda_or_rocm([
        ":gpu_utils",
        ":gpu_prim_hdrs",
    ]),
)

tf_kernel_library(
    name = "depthwise_conv_grad_op",
    hdrs = [
        "depthwise_conv_op.h",
    ],
    features = ["-layering_check"],
    prefix = "depthwise_conv_grad_op",
    deps = [
        ":cast_op",
        ":conv_ops",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/util:determinism_for_kernels",
    ] + if_cuda([
        "@local_config_cuda//cuda:cudnn_header",
    ]),
)

cc_library(
    name = "nn",
    deps = [
        ":batch_norm_op",
        ":bias_op",
        ":conv_ops",
        ":data_format_ops",
        ":depthwise_conv_grad_op",
        ":depthwise_conv_op",
        ":dilation_ops",
        ":fused_batch_norm_op",
        ":in_topk_op",
        ":l2loss_op",
        ":lrn_op",
        ":nth_element_op",
        ":relu_op",
        ":softmax_op",
        ":softplus_op",
        ":softsign_op",
        ":topk_op",
        ":xent_op",
    ],
)

# Kernels for the nodes intented to be added to the graph by the Grappler optimizers.
cc_library(
    name = "grappler",
    deps = [
        ":unary_ops_composition",
    ],
)

NN_DEPS = if_cuda_or_rocm([":conv_2d"]) + [
    "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel",
    ":ops_util",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:nn_grad",
    "//tensorflow/core/framework:bounds_check",
    "@eigen_archive//:eigen3",
]

tf_kernel_library(
    name = "batch_norm_op",
    prefix = "batch_norm_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "data_format_ops",
    prefix = "data_format_ops",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "bias_op",
    features = ["-layering_check"],
    prefix = "bias_op",
    deps = NN_DEPS + [
        ":loose_headers",
        ":redux_functor",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "//tensorflow/core/util:determinism_for_kernels",
    ] + if_cuda_or_rocm([
        ":reduction_ops",
        "@local_xla//xla/stream_executor:event_based_timer",
    ]) + if_cuda([
        "@local_config_cuda//cuda:cub_headers",
        "//tensorflow/core/platform:stream_executor",
    ]) + if_rocm([
        "@local_config_rocm//rocm:rocprim",
    ]),
)

tf_kernel_library(
    name = "fused_batch_norm_op",
    features = ["-layering_check"],
    prefix = "fused_batch_norm_op",
    deps = NN_DEPS + [
        ":cast_op",
        ":fill_functor",
        ":redux_functor",
        ":transpose_functor",
        "//tensorflow/core/util:determinism_for_kernels",
    ] + if_cuda([
        "//tensorflow/core/platform:stream_executor",
    ]),
)

tf_kernel_library(
    name = "in_topk_op",
    features = if_cuda(["-layering_check"]),
    prefix = "in_topk_op",
    deps = NN_DEPS + [":reduction_ops"],
)

tf_kernel_library(
    name = "lrn_op",
    features = ["-layering_check"],
    prefix = "lrn_op",
    deps = NN_DEPS + if_rocm([":conv_ops_gpu_hdrs"]) + [":loose_headers"],
)

tf_kernel_library(
    name = "relu_op",
    copts = if_mlir_generated_experimental_kernels_enabled(
        ["-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED"],
    ) + if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    features = if_cuda(["-layering_check"]),
    # *.cu.cc sources are compiled with gpu_copts instead of copts.
    gpu_copts = if_mlir_generated_experimental_kernels_enabled(
        ["-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED"],
    ) + if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    prefix = "relu_op",
    deps = NN_DEPS + [
        "//tensorflow/core/kernels/mlir_generated:relu_op",
    ],
)

tf_kernel_library(
    name = "softmax_op",
    prefix = "softmax_op",
    deps = NN_DEPS + if_cuda_or_rocm([
        ":reduction_ops",
    ]) + [
        ":gpu_prim_hdrs",
        ":loose_headers",
    ],
)

tf_kernel_library(
    name = "softplus_op",
    copts = if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    # *.cu.cc sources are compiled with gpu_copts instead of copts.
    gpu_copts = if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    prefix = "softplus_op",
    deps = NN_DEPS + [
        "//tensorflow/core/kernels/mlir_generated:softplus_op",
    ],
)

tf_kernel_library(
    name = "softsign_op",
    copts = if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    # *.cu.cc sources are compiled with gpu_copts instead of copts.
    gpu_copts = if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    prefix = "softsign_op",
    deps = NN_DEPS + [
        "//tensorflow/core/kernels/mlir_generated:softsign_op",
    ],
)

tf_kernel_library(
    name = "topk_op",
    srcs = ["topk_op.cc"],
    hdrs = ["topk_op.h"],
    gpu_srcs = [
        "topk_op.h",
        "topk_op_gpu.h",
        "topk_op_gpu_bfloat16.cu.cc",
        "topk_op_gpu_double.cu.cc",
        "topk_op_gpu_float.cu.cc",
        "topk_op_gpu_half.cu.cc",
        "topk_op_gpu_uint64.cu.cc",
        "topk_op_gpu_int64.cu.cc",
        "topk_op_gpu_uint32.cu.cc",
        "topk_op_gpu_int32.cu.cc",
        "topk_op_gpu_int16.cu.cc",
        "topk_op_gpu_uint16.cu.cc",
        "topk_op_gpu_int8.cu.cc",
        "topk_op_gpu_uint8.cu.cc",
    ],
    deps = NN_DEPS + [
        ":gpu_prim_hdrs",
        ":gpu_prim_helpers",
    ],
)

tf_kernel_library(
    name = "nth_element_op",
    prefix = "nth_element_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "xent_op",
    gpu_copts = tf_disable_ptxas_warning_flags(),
    prefix = "xent_op",
    deps = NN_DEPS + ["//tensorflow/core/util:determinism_for_kernels"],
)

tf_kernel_library(
    name = "bincount_op",
    prefix = "bincount_op",
    deps = [
        ":fill_functor",
        ":gpu_prim_hdrs",
        ":sparse_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "histogram_op",
    prefix = "histogram_op",
    deps = [
        ":gpu_prim_hdrs",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "l2loss_op",
    features = if_cuda(["-layering_check"]),
    prefix = "l2loss_op",
    deps = [
        ":gpu_prim_hdrs",
        ":reduction_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:nn_grad",
        "@eigen_archive//:eigen3",
    ],
)

tf_cuda_cc_test(
    name = "lrn_op_test",
    srcs = ["lrn_op_test.cc"],
    deps = [
        ":nn",
        ":ops_testutil",
        ":ops_util",
        ":xent_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "xent_op_test",
    srcs = ["xent_op_test.cc"],
    deps = [
        ":nn",
        ":ops_testutil",
        ":ops_util",
        ":xent_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "nn_ops_test",
    srcs = ["nn_ops_test.cc"],
    deps = [
        ":host_constant_op",
        ":nn",
        ":ops_testutil",
        ":ops_util",
        ":pooling_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "pooling_ops",
    srcs = [
        "avgpooling_op.cc",
        "cudnn_pooling_gpu.cc",
        "fractional_avg_pool_op.cc",
        "fractional_max_pool_op.cc",
        "fractional_pool_common.cc",
        "maxpooling_op.cc",
        "pooling_ops_3d.cc",
        "pooling_ops_common.cc",
    ],
    hdrs = [
        "avgpooling_op.h",
        "cudnn_pooling_gpu.h",
        "fractional_pool_common.h",
        "maxpooling_op.h",
        "pooling_ops_3d.h",
        "pooling_ops_common.h",
    ],
    features = ["-layering_check"],
    gpu_srcs = [
        "avgpooling_op.h",
        "avgpooling_op_gpu.cu.cc",
        "maxpooling_op.h",
        "maxpooling_op_gpu.cu.cc",
        "maxpooling_op_gpu.h",
        "pooling_ops_common.h",
        "pooling_ops_common_gpu.h",
        "pooling_ops_3d_gpu.h",
        "pooling_ops_3d_gpu.cu.cc",
    ],
    deps = [
        ":cast_op",
        ":conv_2d",
        ":conv_3d",
        ":conv_ops",
        ":eigen_helpers",
        ":loose_headers",
        ":numeric_options_utils",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/platform:stream_executor",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "fake_quant_ops",
    srcs = ["fake_quant_ops.cc"],
    hdrs = ["fake_quant_ops_functor.h"],
    gpu_copts = tf_disable_ptxas_warning_flags(),
    gpu_srcs = [
        "fake_quant_ops_gpu.cu.cc",
        "fake_quant_ops_functor.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util:determinism_for_kernels",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pooling_ops_hdrs",
    hdrs = [
        "avgpooling_op.h",
        "maxpooling_op.h",
        "pooling_ops_common.h",
    ],
    deps = [
        ":eigen_helpers",
        ":ops_util_hdrs",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "dilation_ops",
    prefix = "dilation_ops",
    deps = [
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util:determinism_for_kernels",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "batch_space_ops",
    srcs = [
        "batchtospace_op.cc",
        "spacetobatch_functor.cc",
        "spacetobatch_functor.h",
        "spacetobatch_op.cc",
    ],
    gpu_srcs = [
        "spacetobatch_functor.h",
        "spacetobatch_functor_gpu.cu.cc",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/util:overflow",
        "@eigen_archive//:eigen3",
    ],
)

tf_cuda_cc_test(
    name = "spacetobatch_benchmark_test",
    srcs = ["spacetobatch_benchmark_test.cc"],
    deps = [
        ":batch_space_ops",
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "depth_space_ops",
    srcs = [
        "depthtospace_op.cc",
        "spacetodepth_op.cc",
    ],
    hdrs = [
        "depthtospace_op.h",
        "spacetodepth_op.h",
    ],
    gpu_srcs = [
        "depthtospace_op.h",
        "depthtospace_op_gpu.cu.cc",
        "spacetodepth_op.h",
        "spacetodepth_op_gpu.cu.cc",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "parsing",
    deps = [
        ":decode_compressed_op",
        ":decode_csv_op",
        ":decode_padded_raw_op",
        ":decode_raw_op",
        ":example_parsing_ops",
        ":parse_tensor_op",
        ":string_to_number_op",
    ],
)

PARSING_DEPS = [
    "@com_google_absl//absl/base",
    "//tensorflow/core:core_cpu_internal",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:protos_all_cc",
]

tf_kernel_library(
    name = "decode_csv_op",
    prefix = "decode_csv_op",
    deps = PARSING_DEPS,
)

tf_kernel_library(
    name = "decode_raw_op",
    prefix = "decode_raw_op",
    deps = PARSING_DEPS,
)

tf_kernel_library(
    name = "decode_padded_raw_op",
    prefix = "decode_padded_raw_op",
    deps = PARSING_DEPS,
)

tf_kernel_library(
    name = "decode_compressed_op",
    prefix = "decode_compressed_op",
    deps = [
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/status",
        "@net_zstd//:zstdlib",
    ] + PARSING_DEPS,
)

tf_kernel_library(
    name = "example_parsing_ops",
    prefix = "example_parsing_ops",
    deps = PARSING_DEPS,
)

tf_kernel_library(
    name = "parse_tensor_op",
    prefix = "parse_tensor_op",
    deps = [
        "//tensorflow/core/util/tensor_bundle:byteswaptensor",
        "@com_google_absl//absl/strings",
    ] + PARSING_DEPS,
)

tf_cc_test(
    name = "parse_tensor_test",
    srcs = ["parse_tensor_test.cc"],
    deps = [
        ":ops_testutil",
        ":parse_tensor_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "string_to_number_op",
    prefix = "string_to_number_op",
    deps = PARSING_DEPS,
)

cc_library(
    name = "random_ops",
    deps = [
        ":random_op",
        ":random_shuffle_op",
    ],
)

RANDOM_OPS_DEPS = [
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
]

cc_library(
    name = "random_ops_util",
    hdrs = ["random_ops_util.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "random_op",
    features = ["-layering_check"],
    prefix = "random_op",
    deps = RANDOM_OPS_DEPS,
)

cc_library(
    name = "shuffle_common",
    hdrs = ["shuffle_common.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "random_shuffle_op",
    prefix = "random_shuffle_op",
    deps = RANDOM_OPS_DEPS + [
        ":shuffle_common",
    ],
)

cc_library(
    name = "stateless_random_ops_v2_header",
    hdrs = ["stateless_random_ops_v2.h"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "stateless_random_ops_v2_util",
    hdrs = ["stateless_random_ops_v2_util.h"],
    deps = [
        ":random_op",
        ":stateless_random_ops_v2_header",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "stochastic_cast_op_header",
    hdrs = ["stochastic_cast_op.h"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

tf_kernel_library(
    name = "stateless_shuffle",
    prefix = "stateless_shuffle",
    deps = [
        ":random_ops_util",
        ":shuffle_common",
        ":stateless_random_ops_v2_util",
        "//tensorflow/core:framework",
    ],
)

tf_cuda_cc_test(
    name = "random_op_test",
    size = "small",
    srcs = ["random_op_test.cc"],
    deps = [
        ":host_constant_op",
        ":random_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "stateful_random_ops_header",
    hdrs = ["stateful_random_ops.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "stateful_random_ops",
    features = if_cuda(["-layering_check"]),
    prefix = "stateful_random_ops",
    deps = [
        ":dense_update_functor",
        ":fill_functor",
        ":gather_functor",
        ":mutex_ops",
        ":random_op",
        ":resource_variable_ops",
        ":scatter_functor",
        ":state",
        ":stateful_random_ops_header",
        ":training_op_helpers",
        ":variable_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:bounds_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

tf_kernel_library(
    name = "stateless_random_gamma_op",
    features = ["-layering_check"],
    prefix = "stateless_random_gamma_op",
    deps = [
        ":stateless_random_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "stateless_random_ops",
    prefix = "stateless_random_ops",
    deps = [
        ":random_op",
        ":random_poisson_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
    ],
)

cc_library(
    name = "random_index_shuffle",
    srcs = ["random_index_shuffle.cc"],
    hdrs = ["random_index_shuffle.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "random_index_shuffle_test",
    srcs = ["random_index_shuffle_test.cc"],
    deps = [
        ":random_index_shuffle",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

tf_kernel_library(
    name = "random_index_shuffle_ops",
    features = ["-layering_check"],
    prefix = "random_index_shuffle_ops",
    deps = [
        ":random_index_shuffle",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "required",
    deps = [
        ":no_op",
        ":sendrecv_ops",
    ],
)

REQUIRED_DEPS = [
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:protos_all_cc",
]

tf_kernel_library(
    name = "no_op",
    prefix = "no_op",
    deps = REQUIRED_DEPS,
)

tf_kernel_library(
    name = "sendrecv_ops",
    prefix = "sendrecv_ops",
    deps = REQUIRED_DEPS + [
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

tf_cc_test(
    name = "sendrecv_ops_test",
    srcs = ["sendrecv_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":sendrecv_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "sparse",
    deps = [
        ":deserialize_sparse_string_op",
        ":deserialize_sparse_variant_op",
        ":serialize_sparse_op",
        ":sparse_add_grad_op",
        ":sparse_add_op",
        ":sparse_concat_op",
        ":sparse_cross_op",
        ":sparse_dense_binary_op_shared",
        ":sparse_fill_empty_rows_op",
        ":sparse_reduce_op",
        ":sparse_reorder_op",
        ":sparse_reshape_op",
        ":sparse_slice_grad_op",
        ":sparse_slice_op",
        ":sparse_softmax",
        ":sparse_sparse_binary_op_shared",
        ":sparse_split_op",
        ":sparse_tensor_dense_add_op",
        ":sparse_tensor_dense_matmul_op",
        ":sparse_tensors_map_ops",
        ":sparse_to_dense_op",
        ":sparse_xent_op",
    ],
)

SPARSE_DEPS = [
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    ":sparse_utils",
]

tf_kernel_library(
    name = "sparse_add_grad_op",
    prefix = "sparse_add_grad_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_add_op",
    prefix = "sparse_add_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_concat_op",
    features = ["-layering_check"],
    prefix = "sparse_concat_op",
    deps = SPARSE_DEPS + if_cuda_or_rocm([
        ":gpu_device_array",
        ":gpu_prim_helpers",
    ]),
)

tf_kernel_library(
    name = "sparse_fill_empty_rows_op",
    prefix = "sparse_fill_empty_rows_op",
    deps = SPARSE_DEPS + [
        "fill_empty_rows_functor",
        ":gpu_prim_hdrs",
    ] + if_cuda_or_rocm([
        ":gpu_prim_helpers",
    ]) + if_cuda([
        "//tensorflow/core/util:cuda_solvers",
    ]) + if_rocm([
        "//tensorflow/core/util:rocm_solvers",
    ]),
)

tf_kernel_library(
    name = "fill_empty_rows_functor",
    features = if_cuda(["-layering_check"]),
    prefix = "fill_empty_rows_functor",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ] + [":gpu_prim_hdrs"] + if_cuda_or_rocm([
        ":gpu_prim_helpers",
    ]) + if_cuda([
        "//tensorflow/core/util:cuda_solvers",
    ]) + if_rocm([
        "//tensorflow/core/util:rocm_solvers",
    ]),
)

tf_kernel_library(
    name = "sparse_cross_op",
    features = ["-layering_check"],
    prefix = "sparse_cross_op",
    deps = SPARSE_DEPS + [
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_reduce_op",
    prefix = "sparse_reduce_op",
    deps = SPARSE_DEPS + ["@com_google_absl//absl/status"],
)

tf_kernel_library(
    name = "sparse_dense_binary_op_shared",
    prefix = "sparse_dense_binary_op_shared",
    deps = SPARSE_DEPS + [
        ":cwise_op",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_sparse_binary_op_shared",
    prefix = "sparse_sparse_binary_op_shared",
    deps = SPARSE_DEPS + [
        ":cwise_op",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_reorder_op",
    features = if_cuda(["-layering_check"]),
    prefix = "sparse_reorder_op",
    deps = SPARSE_DEPS + if_cuda_or_rocm([
        ":gpu_prim_hdrs",
        ":gpu_prim_helpers",
    ]),
)

tf_kernel_library(
    name = "sparse_reshape_op",
    prefix = "sparse_reshape_op",
    deps = SPARSE_DEPS + [
        ":reshape_util",
    ],
)

tf_kernel_library(
    name = "sparse_slice_grad_op",
    features = if_cuda(["-layering_check"]),
    prefix = "sparse_slice_grad_op",
    deps = SPARSE_DEPS + if_cuda_or_rocm([
        ":gpu_prim_hdrs",
    ]),
)

tf_kernel_library(
    name = "sparse_slice_op",
    features = if_cuda(["-layering_check"]),
    prefix = "sparse_slice_op",
    deps = SPARSE_DEPS + if_cuda_or_rocm([
        ":gpu_device_array",
        ":gpu_prim_helpers",
    ]) + if_cuda([
        "//tensorflow/core/util:cuda_solvers",
    ]) + if_rocm([
        "//tensorflow/core/util:rocm_solvers",
    ]),
)

tf_kernel_library(
    name = "sparse_softmax",
    prefix = "sparse_softmax",
    deps = SPARSE_DEPS + [
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_split_op",
    features = if_cuda(["-layering_check"]),
    prefix = "sparse_split_op",
    deps = SPARSE_DEPS + if_cuda_or_rocm([
        ":gpu_device_array",
        ":gpu_prim_helpers",
    ]) + if_cuda([
        "//tensorflow/core/util:cuda_solvers",
    ]) + if_rocm([
        "//tensorflow/core/util:rocm_solvers",
    ]),
)

tf_kernel_library(
    name = "sparse_tensor_dense_add_op",
    prefix = "sparse_tensor_dense_add_op",
    deps = SPARSE_DEPS + [
        ":scatter_functor",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_tensor_dense_matmul_op",
    prefix = "sparse_tensor_dense_matmul_op",
    deps = SPARSE_DEPS + [
        ":fill_functor",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/util:determinism_for_kernels",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_to_dense_op",
    features = ["-layering_check"],
    prefix = "sparse_to_dense_op",
    deps = SPARSE_DEPS + [
        ":loose_headers",
        "@eigen_archive//:eigen3",
    ] + if_cuda_or_rocm([
        ":gpu_utils",
        "//tensorflow/core/platform:stream_executor",
    ]),
)

tf_kernel_library(
    name = "sparse_xent_op",
    features = if_cuda(["-layering_check"]),
    gpu_copts = tf_disable_ptxas_warning_flags(),
    prefix = "sparse_xent_op",
    deps = SPARSE_DEPS + [
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/util:determinism_for_kernels",
        "@eigen_archive//:eigen3",
    ] + if_cuda_or_rocm([
        ":reduction_ops",
    ]) + if_cuda([
        "@local_config_cuda//cuda:cub_headers",
    ]) + if_rocm([
        "@local_config_rocm//rocm:rocprim",
    ]),
)

tf_kernel_library(
    name = "serialize_sparse_op",
    prefix = "serialize_sparse_op",
    deps = SPARSE_DEPS + [
        ":reshape_util",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "deserialize_sparse_string_op",
    prefix = "deserialize_sparse_string_op",
    deps = SPARSE_DEPS + [
        ":reshape_util",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "deserialize_sparse_variant_op",
    prefix = "deserialize_sparse_variant_op",
    deps = SPARSE_DEPS + [
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "sparse_tensors_map_ops",
    features = ["-layering_check"],
    prefix = "sparse_tensors_map_ops",
    deps = SPARSE_DEPS,
)

tf_cuda_cc_tests(
    name = "sparse2_tests",
    size = "small",
    srcs = [
        "sparse_fill_empty_rows_op_test.cc",
        "sparse_tensor_dense_matmul_op_test.cc",
        "sparse_to_dense_op_test.cc",
        "sparse_xent_op_test.cc",
    ],
    deps = [
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        ":sparse",
        ":xent_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:status_matchers",
    ],
)

cc_library(
    name = "loss_updaters",
    hdrs = [
        "hinge-loss.h",
        "logistic-loss.h",
        "loss.h",
        "poisson-loss.h",
        "smooth-hinge-loss.h",
        "squared-loss.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "loss_test",
    size = "small",
    srcs = ["loss_test.cc"],
    deps = [
        ":loss_updaters",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "sdca_ops_test",
    size = "small",
    srcs = ["sdca_ops_test.cc"],
    deps = [
        ":ops_util",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "sdca_ops",
    prefix = "sdca_ops",
    deps = [
        ":loss_updaters",
        ":sdca_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sdca_internal",
    srcs = ["sdca_internal.cc"],
    hdrs = ["sdca_internal.h"],
    deps = [
        ":loss_updaters",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel",
    ],
)

cc_library(
    name = "state",
    deps = [
        ":count_up_to_op",
        ":dense_update_ops",
        ":scatter_nd_op",
        ":scatter_op",
        ":variable_ops",
    ],
)

STATE_DEPS = [
    ":assign_op",
    "//tensorflow/core/framework:bounds_check",
    ":fill_functor",
    ":scatter_functor",
    "@eigen_archive//:eigen3",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
]

tf_kernel_library(
    name = "count_up_to_op",
    prefix = "count_up_to_op",
    deps = STATE_DEPS + [":variable_ops"],
)

tf_kernel_library(
    name = "dense_update_ops",
    prefix = "dense_update_ops",
    deps = STATE_DEPS + [":dense_update_functor"],
)

tf_kernel_library(
    name = "scatter_op",
    prefix = "scatter_op",
    deps = STATE_DEPS + [
        ":loose_headers",
        "//tensorflow/core/util:determinism_for_kernels",
    ],
)

tf_kernel_library(
    name = "count_ops",
    prefix = "count_ops",
    deps = STATE_DEPS + [
        "//tensorflow/core/framework:op_requires",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "count_ops_test",
    size = "small",
    srcs = ["count_ops_test.cc"],
    deps = [
        ":count_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "scatter_nd_util",
    srcs = ["scatter_nd_util.cc"],
    hdrs = ["scatter_nd_util.h"],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

tf_kernel_library(
    name = "scatter_nd_op",
    srcs = [
        "scatter_nd_op.cc",
        "scatter_nd_op_cpu_impl_0.cc",
        "scatter_nd_op_cpu_impl_1.cc",
        "scatter_nd_op_cpu_impl_2.cc",
        "scatter_nd_op_cpu_impl_3.cc",
        "scatter_nd_op_cpu_impl_4.cc",
        "scatter_nd_op_cpu_impl_5.cc",
        "scatter_nd_op_cpu_impl_6.cc",
        "scatter_nd_op_cpu_impl_7.cc",
    ],
    hdrs = [
        "scatter_nd_op.h",
    ],
    features = ["-layering_check"],
    gpu_copts = if_not_windows([
        "-Wno-pass-failed",  # clang misses #pragma loop optimizations
    ]),
    gpu_srcs = [
        "scatter_nd_op.h",
        "scatter_nd_op_gpu.cu.cc",
    ],
    textual_hdrs = [
        "scatter_nd_op_cpu_impl.h",
    ],
    deps = STATE_DEPS + [
        ":dense_update_functor",
        ":inplace_ops",
        ":scatter_nd_util",
        ":training_op_helpers",
        ":variable_ops",
    ],
)

tf_kernel_library(
    name = "variable_ops",
    features = ["-layering_check"],
    prefix = "variable_ops",
    deps = STATE_DEPS,
)

tf_kernel_library(
    name = "mutex_ops",
    prefix = "mutex_ops",
    deps = STATE_DEPS + [":ops_util"],
)

tf_cc_test(
    name = "scatter_op_test",
    size = "small",
    srcs = ["scatter_op_test.cc"],
    deps = [
        ":fill_functor",
        ":ops_testutil",
        ":ops_util",
        ":scatter_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "scatter_nd_op_test",
    size = "small",
    srcs = ["scatter_nd_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":scatter_nd_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "string",
    deps = [
        ":as_string_op",
        ":base64_ops",
        ":reduce_join_op",
        ":regex_full_match_op",
        ":regex_replace_op",
        ":string_format_op",
        ":string_join_op",
        ":string_length_op",
        ":string_lower_op",
        ":string_ngrams_op",
        ":string_split_op",
        ":string_strip_op",
        ":string_to_hash_bucket_op",
        ":string_upper_op",
        ":substr_op",
        ":tensor_to_hash_bucket_op",
        ":unicode_ops",
        ":unicode_script_op",
        ":unsorted_segment_join_op",
    ],
)

cc_library(
    name = "string_util",
    srcs = ["string_util.cc"],
    hdrs = ["string_util.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@icu//:common",
    ],
)

STRING_DEPS = [
    "//tensorflow/core/framework:bounds_check",
    ":string_util",
    "@eigen_archive//:eigen3",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
]

tf_kernel_library(
    name = "string_to_hash_bucket_op",
    srcs = [
        "string_to_hash_bucket_fast_op.cc",
        "string_to_hash_bucket_op.cc",
    ],
    hdrs = [
        "string_to_hash_bucket_fast_op.h",
        "string_to_hash_bucket_op.h",
    ],
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "tensor_to_hash_bucket_op",
    prefix = "tensor_to_hash_bucket_op",
    deps = STRING_DEPS + if_oss(
        if_cuda_or_rocm(["@farmhash_gpu_archive//:farmhash_gpu"]),
        tf_fingerprint_deps(),
    ),
)

tf_kernel_library(
    name = "reduce_join_op",
    prefix = "reduce_join_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "unsorted_segment_join_op",
    prefix = "unsorted_segment_join_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "string_format_op",
    prefix = "string_format_op",
    deps = STRING_DEPS + ["@com_google_absl//absl/strings"],
)

tf_cc_test(
    name = "string_format_op_test",
    size = "small",
    srcs = ["string_format_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":string_format_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "string_join_op",
    prefix = "string_join_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "string_length_op",
    prefix = "string_length_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "regex_full_match_op",
    prefix = "regex_full_match_op",
    deps = STRING_DEPS + ["@com_googlesource_code_re2//:re2"],
)

tf_kernel_library(
    name = "regex_replace_op",
    prefix = "regex_replace_op",
    deps = STRING_DEPS + ["@com_googlesource_code_re2//:re2"],
)

tf_cc_test(
    name = "regex_replace_op_test",
    size = "small",
    srcs = ["regex_replace_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":regex_replace_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "string_split_op",
    prefix = "string_split_op",
    deps = STRING_DEPS,
)

tf_cc_test(
    name = "string_split_op_test",
    size = "small",
    srcs = ["string_split_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":string_split_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "string_ngrams_op",
    srcs = ["string_ngrams_op.cc"],
    deps = STRING_DEPS + [
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "string_ngrams_op_test",
    srcs = ["string_ngrams_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":string_ngrams_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "string_strip_op",
    prefix = "string_strip_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "string_lower_op",
    prefix = "string_lower_op",
    deps = STRING_DEPS + [
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

tf_kernel_library(
    name = "string_upper_op",
    prefix = "string_upper_op",
    deps = STRING_DEPS + [
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

tf_kernel_library(
    name = "substr_op",
    prefix = "substr_op",
    deps = STRING_DEPS,
)

tf_cc_test(
    name = "substr_op_test",
    size = "small",
    srcs = ["substr_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":substr_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "as_string_op",
    features = ["-layering_check"],
    prefix = "as_string_op",
    deps = STRING_DEPS,
)

tf_cc_test(
    name = "as_string_op_test",
    size = "small",
    srcs = ["as_string_op_test.cc"],
    deps = [
        ":as_string_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "unicode_ops",
    prefix = "unicode_ops",
    deps = [
        ":string_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:bounds_check",
        "//third_party/icu/data:conversion_data",
        "@eigen_archive//:eigen3",
        "@icu//:common",
    ],
)

tf_kernel_library(
    name = "base64_ops",
    prefix = "base64_ops",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "training_ops",
    prefix = "training_ops",
    deps = [
        ":training_op_helpers",
        ":variable_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "training_ops_test",
    size = "small",
    srcs = ["training_ops_test.cc"],
    deps = [
        ":dense_update_ops",
        ":ops_util",
        ":training_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "multinomial_op",
    features = if_cuda(["-layering_check"]),
    prefix = "multinomial_op",
    deps = [
        ":gpu_prim_hdrs",
        ":random_op",
        ":random_ops",
        ":stateless_random_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ] + if_cuda_or_rocm([
        ":reduction_ops",
    ]),
)

tf_cuda_cc_test(
    name = "multinomial_op_test",
    size = "small",
    srcs = ["multinomial_op_test.cc"],
    deps = [
        ":host_constant_op",
        ":multinomial_op",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "parameterized_truncated_normal_op",
    features = if_cuda(["-layering_check"]),
    gpu_copts = if_not_windows([
        "-Wno-pass-failed",  # clang misses #pragma loop optimizations
    ]),
    prefix = "parameterized_truncated_normal_op",
    deps = [
        ":stateless_random_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cuda_cc_test(
    name = "parameterized_truncated_normal_op_test",
    size = "small",
    srcs = ["parameterized_truncated_normal_op_test.cc"],
    deps = [
        ":host_constant_op",
        ":ops_util",
        ":parameterized_truncated_normal_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "random_binomial_op",
    prefix = "random_binomial_op",
    deps = [
        ":cwise_op",
        ":random_op",
        ":resource_variable_ops",
        ":stateful_random_ops",
        ":stateless_random_ops",
        ":training_op_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:random_ops_op_lib",
    ],
)

tf_cuda_cc_test(
    name = "random_binomial_op_test",
    size = "small",
    srcs = ["random_binomial_op_test.cc"],
    deps = [
        ":ops_util",
        ":random_binomial_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "random_poisson_op",
    prefix = "random_poisson_op",
    deps = [
        ":random_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cuda_cc_test(
    name = "random_poisson_op_test",
    size = "small",
    srcs = ["random_poisson_op_test.cc"],
    deps = [
        ":ops_util",
        ":random_poisson_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "word2vec_kernels",
    prefix = "word2vec_kernels",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:word2vec_ops",
    ],
)

tf_kernel_library(
    name = "encode_wav_op",
    prefix = "encode_wav_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:bounds_check",
    ],
)

tf_cc_test(
    name = "encode_wav_op_test",
    size = "small",
    srcs = ["encode_wav_op_test.cc"],
    deps = [
        ":decode_wav_op",
        ":encode_wav_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "decode_wav_op",
    prefix = "decode_wav_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "decode_wav_op_test",
    size = "small",
    srcs = ["decode_wav_op_test.cc"],
    deps = [
        ":decode_wav_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

alias(
    name = "spectrogram_test_data",
    actual = "//tensorflow/core/kernels/spectrogram_test_data:spectrogram_test_data",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "spectrogram",
    srcs = ["spectrogram.cc"],
    hdrs = ["spectrogram.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/fft2d:fft2d_headers",
        "@fft2d",
    ],
)

cc_library(
    name = "spectrogram_test_utils",
    testonly = 1,
    srcs = ["spectrogram_test_utils.cc"],
    hdrs = ["spectrogram_test_utils.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
    ],
)

tf_cc_binary(
    name = "spectrogram_convert_test_data",
    testonly = 1,
    srcs = ["spectrogram_convert_test_data.cc"],
    deps = [
        ":spectrogram_test_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "spectrogram_test",
    size = "medium",
    srcs = ["spectrogram_test.cc"],
    data = [":spectrogram_test_data"],
    deps = [
        ":spectrogram",
        ":spectrogram_test_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:resource_loader",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "spectrogram_op",
    prefix = "spectrogram_op",
    deps = [
        ":spectrogram",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "spectrogram_op_test",
    size = "small",
    srcs = ["spectrogram_op_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":ops_util",
        ":spectrogram_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:status_matchers",
    ],
)

cc_library(
    name = "mfcc_dct",
    srcs = ["mfcc_dct.cc"],
    hdrs = ["mfcc_dct.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "mfcc_dct_test",
    size = "small",
    srcs = ["mfcc_dct_test.cc"],
    deps = [
        ":mfcc_dct",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "mfcc_mel_filterbank",
    srcs = ["mfcc_mel_filterbank.cc"],
    hdrs = ["mfcc_mel_filterbank.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "mfcc_mel_filterbank_test",
    size = "small",
    srcs = ["mfcc_mel_filterbank_test.cc"],
    deps = [
        ":mfcc_mel_filterbank",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "mfcc",
    srcs = ["mfcc.cc"],
    hdrs = ["mfcc.h"],
    copts = tf_copts(),
    deps = [
        ":mfcc_dct",
        ":mfcc_mel_filterbank",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "mfcc_test",
    size = "small",
    srcs = ["mfcc_test.cc"],
    deps = [
        ":mfcc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "mfcc_op",
    prefix = "mfcc_op",
    deps = [
        ":mfcc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "mfcc_op_test",
    size = "small",
    srcs = ["mfcc_op_test.cc"],
    deps = [
        ":mfcc_op",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "audio",
    deps = [
        ":decode_wav_op",
        ":encode_wav_op",
        ":mfcc_op",
        ":spectrogram_op",
    ],
)

cc_library(
    name = "meta_support",
    srcs = ["meta_support.cc"],
    hdrs = ["meta_support.h"],
    deps = [
        ":quantization_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:mutex",
        "@gemmlowp",
    ],
)

# Android libraries -----------------------------------------------------------
filegroup(
    name = "mobile_srcs",
    srcs = [
        "avgpooling_op.h",
        "batch_util.h",
        "cwise_ops.h",
        "cwise_ops_common.h",
        "cwise_ops_gradients.h",
        "eigen_activations.h",
        "eigen_attention.h",
        "eigen_backward_cuboid_convolutions.h",
        "eigen_backward_spatial_convolutions.h",
        "eigen_cuboid_convolution.h",
        "eigen_pooling.h",
        "fifo_queue.h",
        "initializable_lookup_table.cc",
        "initializable_lookup_table.h",
        "lookup_util.cc",
        "lookup_util.h",
        "maxpooling_op.h",
        "ops_util.h",
        "padding_fifo_queue.h",
        "pooling_ops_common.cc",
        "pooling_ops_common.h",
        "queue_base.h",
        "queue_op.h",
        "typed_queue.h",
        "@local_xla//xla/tsl/framework/convolution:eigen_convolution_helpers.h",
        "@local_xla//xla/tsl/framework/convolution:eigen_spatial_convolutions.h",
        "@local_xla//xla/tsl/framework/convolution:eigen_spatial_convolutions-inl.h",
    ],
)

alias(
    name = "android_srcs",
    actual = ":mobile_srcs",
)

# Core kernels we want on Android. Only a subset of kernels to keep
# base library small.
filegroup(
    name = "portable_core_ops",
    srcs = [
        "aggregate_ops.cc",
        "aggregate_ops.h",
        "aggregate_ops_cpu.h",
        "assign_op.h",
        "bias_op.cc",
        "bias_op.h",
        "cast_op.cc",
        "cast_op.h",
        "cast_op_impl.h",
        "cast_op_impl_bfloat.cc",
        "cast_op_impl_bool.cc",
        "cast_op_impl_complex128.cc",
        "cast_op_impl_complex64.cc",
        "cast_op_impl_double.cc",
        "cast_op_impl_float.cc",
        "cast_op_impl_float8.cc",
        "cast_op_impl_half.cc",
        "cast_op_impl_int16.cc",
        "cast_op_impl_int32.cc",
        "cast_op_impl_int4.cc",
        "cast_op_impl_int64.cc",
        "cast_op_impl_int8.cc",
        "cast_op_impl_uint16.cc",
        "cast_op_impl_uint32.cc",
        "cast_op_impl_uint4.cc",
        "cast_op_impl_uint64.cc",
        "cast_op_impl_uint8.cc",
        "concat_lib.h",
        "concat_lib_cpu.cc",
        "concat_lib_cpu.h",
        "concat_op.cc",
        "constant_op.cc",
        "constant_op.h",
        "cwise_ops.h",
        "cwise_ops_common.cc",
        "cwise_ops_common.h",
        "cwise_ops_gradients.h",
        "dense_update_functor.cc",
        "dense_update_functor.h",
        "dense_update_ops.cc",
        "example_parsing_ops.cc",
        "fill_functor.cc",
        "fill_functor.h",
        "function_ops.cc",
        "function_ops.h",
        "gather_functor.h",
        "gather_functor_batched.h",
        "gather_nd_op.cc",
        "gather_nd_op.h",
        "gather_nd_op_cpu_impl_0.cc",
        "gather_nd_op_cpu_impl_1.cc",
        "gather_nd_op_cpu_impl_2.cc",
        "gather_nd_op_cpu_impl_3.cc",
        "gather_nd_op_cpu_impl_4.cc",
        "gather_nd_op_cpu_impl_5.cc",
        "gather_nd_op_cpu_impl_6.cc",
        "gather_nd_op_cpu_impl_7.cc",
        "gather_op.cc",
        "identity_n_op.cc",
        "identity_n_op.h",
        "identity_op.cc",
        "identity_op.h",
        "immutable_constant_op.cc",
        "immutable_constant_op.h",
        "matmul_op_impl.h",
        "matmul_op_real.cc",
        "no_op.cc",
        "no_op.h",
        "one_hot_op.cc",
        "one_hot_op.h",
        "ops_util.h",
        "pack_op.cc",
        "pooling_ops_common.h",
        "redux_functor.h",
        "reshape_op.cc",
        "reshape_op.h",
        "reverse_sequence_op.cc",
        "reverse_sequence_op.h",
        "sendrecv_ops.cc",
        "sendrecv_ops.h",
        "sequence_ops.cc",
        "sequence_ops.h",
        "shape_ops.cc",
        "shape_ops.h",
        "slice_op.cc",
        "slice_op.h",
        "slice_op_cpu_impl_1.cc",
        "slice_op_cpu_impl_2.cc",
        "slice_op_cpu_impl_3.cc",
        "slice_op_cpu_impl_4.cc",
        "slice_op_cpu_impl_5.cc",
        "slice_op_cpu_impl_6.cc",
        "slice_op_cpu_impl_7.cc",
        "slice_op_cpu_impl_8.cc",
        "softmax_op.cc",
        "softmax_op_functor.h",
        "sparse_concat_op.cc",
        "sparse_concat_op.h",
        "split_lib.h",
        "split_lib_cpu.cc",
        "split_op.cc",
        "split_v_op.cc",
        "strided_slice_op.cc",
        "strided_slice_op.h",
        "strided_slice_op_inst_0.cc",
        "strided_slice_op_inst_1.cc",
        "strided_slice_op_inst_2.cc",
        "strided_slice_op_inst_3.cc",
        "strided_slice_op_inst_4.cc",
        "strided_slice_op_inst_5.cc",
        "strided_slice_op_inst_6.cc",
        "strided_slice_op_inst_7.cc",
        "strided_slice_op_inst_8.cc",
        "unpack_op.cc",
        "variable_ops.cc",
        "variable_ops.h",
        "variant_ops_util.cc",
        "variant_ops_util.h",
    ] + [
        "//tensorflow/c/kernels:portable_all_op_kernels",
        "//tensorflow/core/kernels/image:non_max_suppression_op.cc",
        "//tensorflow/core/kernels/image:non_max_suppression_op.h",
        "//tensorflow/core/util:bad_indices_policy.cc",
        "//tensorflow/core/util:bad_indices_policy.h",
    ],
)

# Other kernels we may want on Android.
#
# The kernels can be consumed as a whole or in two groups for
# supporting separate compilation. Note that the split into groups
# is entirely for improving compilation time, and not for
# organizational reasons; you should not depend on any
# of those groups independently.
filegroup(
    name = "portable_extended_ops",
    srcs = [
        ":portable_extended_ops_group1",
        ":portable_extended_ops_group2",
        ":portable_quantized_ops",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "portable_extended_ops_headers",
    srcs = [
        "argmax_op.h",
        "avgpooling_op.h",
        "batch_norm_op.h",
        "bincount_op.h",
        "broadcast_to_op.h",
        "bucketize_op.h",
        "checkpoint_callback_manager.h",
        "concat_lib.h",
        "control_flow_ops.h",
        "conv_2d.h",
        "conv_3d.h",
        "conv_ops.h",
        "conv_ops_gpu.h",
        "conv_ops_impl.h",
        "data_format_ops.h",
        "depthtospace_op.h",
        "depthwise_conv_op.h",
        "diag_op.h",
        "dilation_ops.h",
        "fake_quant_ops_functor.h",
        "fill_empty_rows_functor.h",
        "function_ops.h",
        "fused_batch_norm_op.h",
        "gpu_utils.h",
        "inplace_ops.cc",
        "inplace_ops_functor.h",
        "l2loss_op.h",
        "list_kernels.h",
        "lookup_table_init_op.h",
        "lookup_table_op.h",
        "map_kernels.h",
        "maxpooling_op.h",
        "mfcc.h",
        "mfcc_dct.h",
        "mfcc_mel_filterbank.h",
        "multinomial_op.h",
        "pad_op.h",
        "partitioned_function_ops.h",
        "pooling_ops_3d.h",
        "ragged_tensor_variant.h",
        "ragged_utils.h",
        "random_index_shuffle.h",
        "random_op.h",
        "random_poisson_op.h",
        "reduction_ops.h",
        "reduction_ops_common.h",
        "relu_op.h",
        "relu_op_functor.h",
        "reshape_util.h",
        "resource_variable_ops.h",
        "resource_variable_util.h",
        "reverse_op.h",
        "roll_op.h",
        "save_restore_tensor.h",
        "scan_ops.h",
        "scatter_functor.h",
        "scatter_nd_op.h",
        "scatter_nd_util.h",
        "searchsorted_op.h",
        "segment_reduction_ops.h",
        "segment_reduction_ops_impl.h",
        "softplus_op.h",
        "softsign_op.h",
        "spacetobatch_functor.h",
        "spacetodepth_op.h",
        "sparse_reorder_op.h",
        "sparse_slice_op.h",
        "sparse_tensor_dense_matmul_op.h",
        "sparse_utils.h",
        "sparse_xent_op.h",
        "spectrogram.h",
        "stateless_random_gamma_op.h",
        "stateless_random_ops.h",
        "stateless_random_ops_v2.h",
        "stochastic_cast_op.h",
        "string_to_hash_bucket_fast_op.h",
        "string_to_hash_bucket_op.h",
        "string_util.h",
        "tensor_array.h",
        "tensor_list.h",
        "tensor_map.h",
        "tile_functor.h",
        "tile_ops_impl.h",
        "topk_op.h",
        "training_op_helpers.h",
        "training_ops.h",
        "transpose_functor.h",
        "transpose_op.h",
        "where_op.h",
        "xent_op.h",
    ] + [
        "//tensorflow/core/kernels/data:portable_all_op_kernels_headers",
        "//tensorflow/core/kernels/image:adjust_contrast_op.h",
        "//tensorflow/core/kernels/image:adjust_hue_op.h",
        "//tensorflow/core/kernels/image:adjust_saturation_op.h",
        "//tensorflow/core/kernels/image:colorspace_op.h",
        "//tensorflow/core/kernels/image:extract_image_patches_op.h",
        "//tensorflow/core/kernels/image:image_ops.h",
        "//tensorflow/core/kernels/image:mirror_pad_op.h",
        "//tensorflow/core/kernels/image:mirror_pad_op_cpu_impl.h",
        "//tensorflow/core/kernels/image:resize_bilinear_op.h",
        "//tensorflow/core/kernels/image:resize_nearest_neighbor_op.h",
        "//tensorflow/core/kernels/linalg:linalg_ops_common.h",
        "//tensorflow/core/kernels/linalg:matrix_band_part_op.h",
        "//tensorflow/core/kernels/linalg:matrix_diag_op.h",
        "//tensorflow/core/kernels/linalg:matrix_set_diag_op.h",
        "//tensorflow/core/kernels/linalg:matrix_triangular_solve_op_impl.h",
        "//tensorflow/core/kernels/linalg:qr_op_impl.h",
        "//tensorflow/core/kernels/uniform_quant_ops:portable_all_op_kernels_headers",
        "//tensorflow/core/util:bad_indices_policy.h",
        "//tensorflow/core/util:image_resizer_state.h",
    ],
)

filegroup(
    name = "portable_extended_ops_group1",
    srcs = [
        "argmax_op.cc",
        "autotune_conv_impl.h",
        "avgpooling_op.cc",
        "batch_norm_op.cc",
        "bcast_ops.cc",
        "check_numerics_op.cc",
        "control_flow_ops.cc",
        "conv_2d.h",
        "conv_grad_filter_ops.cc",
        "conv_grad_filter_ops_3d.cc",
        "conv_grad_filter_ops_launcher.cc",
        "conv_grad_input_ops.cc",
        "conv_grad_input_ops.h",
        "conv_grad_input_ops_3d.cc",
        "conv_grad_input_ops_double.cc",
        "conv_grad_input_ops_float.cc",
        "conv_grad_input_ops_half.cc",
        "conv_grad_input_ops_int32.cc",
        "conv_grad_ops.h",
        "conv_grad_shape_utils.cc",
        "conv_grad_shape_utils.h",
        "conv_ops.cc",
        "conv_ops_3d.cc",
        "conv_ops_bfloat16.cc",
        "conv_ops_double.cc",
        "conv_ops_float.cc",
        "conv_ops_fused_double.cc",
        "conv_ops_fused_float.cc",
        "conv_ops_fused_half.cc",
        "conv_ops_fused_image_transform.cc",
        "conv_ops_fused_impl.h",
        "conv_ops_fused_int8.cc",
        "conv_ops_half.cc",
        "conv_ops_int32.cc",
        "conv_ops_using_gemm.cc",
        "cwise_op_abs.cc",
        "cwise_op_add_1.cc",
        "cwise_op_add_2.cc",
        "cwise_op_arg.cc",
        "cwise_op_atan.cc",
        "cwise_op_atan2.cc",
        "cwise_op_bitwise_and.cc",
        "cwise_op_bitwise_or.cc",
        "cwise_op_bitwise_xor.cc",
        "cwise_op_ceil.cc",
        "cwise_op_clip.cc",
        "cwise_op_clip.h",
        "cwise_op_complex.cc",
        "cwise_op_conj.cc",
        "cwise_op_cos.cc",
        "cwise_op_cosh.cc",
        "cwise_op_div.cc",
        "cwise_op_equal_to_1.cc",
        "cwise_op_equal_to_2.cc",
        "cwise_op_erf.cc",
        "cwise_op_exp.cc",
        "cwise_op_floor.cc",
        "cwise_op_floor_div.cc",
        "cwise_op_floor_mod.cc",
        "cwise_op_greater.cc",
        "cwise_op_greater_equal.cc",
        "cwise_op_imag.cc",
        "cwise_op_invert.cc",
        "cwise_op_isfinite.cc",
        "cwise_op_isnan.cc",
        "cwise_op_leakyrelu.cc",
        "cwise_op_left_shift.cc",
        "cwise_op_less.cc",
        "cwise_op_less_equal.cc",
        "cwise_op_log.cc",
        "cwise_op_logical_and.cc",
        "cwise_op_logical_not.cc",
        "cwise_op_logical_or.cc",
        "cwise_op_maximum.cc",
        "cwise_op_minimum.cc",
        "cwise_op_mul_1.cc",
        "cwise_op_mul_2.cc",
        "cwise_op_neg_1.cc",
        "cwise_op_neg_2.cc",
        "cwise_op_not_equal_to_1.cc",
        "cwise_op_not_equal_to_2.cc",
        "cwise_op_pow.cc",
        "cwise_op_real.cc",
        "cwise_op_reciprocal.cc",
        "cwise_op_right_shift.cc",
        "cwise_op_round.cc",
        "cwise_op_rsqrt.cc",
        "cwise_op_select.cc",
        "cwise_op_sigmoid.cc",
        "cwise_op_sign.cc",
        "cwise_op_sin.cc",
        "cwise_op_sinh.cc",
        "cwise_op_sqrt.cc",
        "cwise_op_square.cc",
        "cwise_op_squared_difference.cc",
        "cwise_op_sub.cc",
        "cwise_op_tan.cc",
        "cwise_op_tanh.cc",
        "cwise_op_xdivy.cc",
        "cwise_op_xlog1py.cc",
        "cwise_op_xlogy.cc",
        "data_format_ops.cc",
        "decode_raw_op.cc",
        "decode_wav_op.cc",
        "deep_conv2d.cc",
        "deep_conv2d.h",
        "depthwise_conv_grad_op.cc",
        "depthwise_conv_op.cc",
        "dynamic_partition_op.cc",
        "encode_wav_op.cc",
        "fake_quant_ops.cc",
        "fifo_queue.cc",
        "fifo_queue_op.cc",
        "fingerprint_op.cc",
        "fused_batch_norm_op.cc",
        "fused_eigen_output_kernels.cc",
        "fused_eigen_output_kernels.h",
        "listdiff_op.cc",
        "population_count_op.cc",
        "population_count_op.h",
        "winograd_transform.h",
        ":portable_extended_ops_headers",
        "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel.cc",
        "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel.h",
    ] + [
        "//tensorflow/core/kernels/image:colorspace_op.cc",
        "//tensorflow/core/kernels/image:crop_and_resize_op.cc",
        "//tensorflow/core/kernels/image:crop_and_resize_op.h",
        "//tensorflow/core/kernels/image:decode_image_op.cc",
        "//tensorflow/core/kernels/image:encode_jpeg_op.cc",
        "//tensorflow/core/kernels/image:encode_png_op.cc",
        "//tensorflow/core/kernels/linalg:einsum_op.h",
        "//tensorflow/core/kernels/linalg:einsum_op_impl.h",
        "//tensorflow/core/kernels/linalg:einsum_op_impl_bfloat16.cc",
        "//tensorflow/core/kernels/linalg:einsum_op_impl_complex128.cc",
        "//tensorflow/core/kernels/linalg:einsum_op_impl_complex64.cc",
        "//tensorflow/core/kernels/linalg:einsum_op_impl_double.cc",
        "//tensorflow/core/kernels/linalg:einsum_op_impl_float.cc",
        "//tensorflow/core/kernels/linalg:einsum_op_impl_half.cc",
        "//tensorflow/core/kernels/linalg:einsum_op_impl_int32.cc",
        "//tensorflow/core/kernels/linalg:einsum_op_impl_int64.cc",
        "//tensorflow/core/kernels/uniform_quant_ops:portable_all_op_kernels",
    ],
)

filegroup(
    name = "portable_extended_ops_group2",
    srcs = [
        "as_string_op.cc",
        "base64_ops.cc",
        "batchtospace_op.cc",
        "bincount_op.cc",
        "broadcast_to_op.cc",
        "bucketize_op.cc",
        "checkpoint_callback_manager.cc",
        "ctc_decoder_ops.cc",
        "decode_padded_raw_op.cc",
        "depthtospace_op.cc",
        "diag_op.cc",
        "dilation_ops.cc",
        "dynamic_stitch_op.cc",
        "fft_ops.cc",
        "functional_ops.cc",
        "in_topk_op.cc",
        "in_topk_op.h",
        "l2loss_op.cc",
        "list_kernels.cc",
        "logging_ops.cc",
        "logging_ops.h",
        "lookup_table_init_op.cc",
        "lookup_table_op.cc",
        "lrn_op.cc",
        "map_kernels.cc",
        "maxpooling_op.cc",
        "mfcc.cc",
        "mfcc_dct.cc",
        "mfcc_mel_filterbank.cc",
        "mfcc_op.cc",
        "multinomial_op.cc",
        "pad_op.cc",
        "padding_fifo_queue.cc",
        "padding_fifo_queue_op.cc",
        "parse_tensor_op.cc",
        "partitioned_function_ops.cc",
        "pooling_ops_3d.cc",
        "queue_base.cc",
        "queue_op.cc",
        "queue_ops.cc",
        "ragged_gather_op.cc",
        "ragged_range_op.cc",
        "ragged_tensor_from_variant_op.cc",
        "ragged_tensor_to_sparse_kernel.cc",
        "ragged_tensor_to_tensor_op.cc",
        "ragged_tensor_to_variant_op.cc",
        "ragged_tensor_variant.cc",
        "random_index_shuffle.cc",
        "random_index_shuffle.h",
        "random_index_shuffle_ops.cc",
        "random_op.cc",
        "random_op_cpu.h",
        "random_ops_util.h",
        "random_poisson_op.cc",
        "random_shuffle_op.cc",
        "reduce_join_op.cc",
        "reduction_ops_all.cc",
        "reduction_ops_any.cc",
        "reduction_ops_common.cc",
        "reduction_ops_max.cc",
        "reduction_ops_mean.cc",
        "reduction_ops_min.cc",
        "reduction_ops_prod.cc",
        "reduction_ops_sum.cc",
        "regex_full_match_op.cc",
        "regex_replace_op.cc",
        "relu_op.cc",
        "reshape_util.cc",
        "resource_variable_ops.cc",
        "resource_variable_util.cc",
        "restore_op.cc",
        "reverse_op.cc",
        "roll_op.cc",
        "save_op.cc",
        "save_restore_tensor.cc",
        "save_restore_v2_ops.cc",
        "scan_ops.cc",
        "scatter_functor.cc",
        "scatter_nd_op.cc",
        "scatter_nd_op_cpu_impl_0.cc",
        "scatter_nd_op_cpu_impl_1.cc",
        "scatter_nd_op_cpu_impl_2.cc",
        "scatter_nd_op_cpu_impl_3.cc",
        "scatter_nd_op_cpu_impl_4.cc",
        "scatter_nd_op_cpu_impl_5.cc",
        "scatter_nd_op_cpu_impl_6.cc",
        "scatter_nd_op_cpu_impl_7.cc",
        "scatter_nd_util.cc",
        "searchsorted_op.cc",
        "segment_reduction_ops_impl_1.cc",
        "segment_reduction_ops_impl_2.cc",
        "segment_reduction_ops_impl_3.cc",
        "segment_reduction_ops_impl_4.cc",
        "segment_reduction_ops_impl_5.cc",
        "session_ops.cc",
        "set_kernels.cc",
        "shuffle_common.h",
        "softplus_op.cc",
        "softsign_op.cc",
        "spacetobatch_functor.cc",
        "spacetobatch_op.cc",
        "spacetodepth_op.cc",
        "sparse_add_op.cc",
        "sparse_cross_op.cc",
        "sparse_fill_empty_rows_op.cc",
        "sparse_reduce_op.cc",
        "sparse_reorder_op.cc",
        "sparse_reshape_op.cc",
        "sparse_slice_op.cc",
        "sparse_tensor_dense_matmul_op.cc",
        "sparse_to_dense_op.cc",
        "sparse_utils.cc",
        "sparse_xent_op.cc",
        "spectrogram.cc",
        "spectrogram_op.cc",
        "stack.cc",
        "stack.h",
        "stack_ops.cc",
        "stateless_random_gamma_op.cc",
        "stateless_random_ops.cc",
        "stateless_random_ops_v2.cc",
        "stateless_random_ops_v2_util.h",
        "stateless_shuffle.cc",
        "stochastic_cast_op.cc",
        "string_format_op.cc",
        "string_join_op.cc",
        "string_length_op.cc",
        "string_lower_op.cc",
        "string_split_op.cc",
        "string_strip_op.cc",
        "string_to_hash_bucket_fast_op.cc",
        "string_to_hash_bucket_op.cc",
        "string_to_number_op.cc",
        "string_util.cc",
        "substr_op.cc",
        "tensor_array.cc",
        "tensor_array_ops.cc",
        "tensor_list.cc",
        "tensor_list_util.cc",
        "tensor_list_util.h",
        "tensor_map.cc",
        "tile_functor_cpu.h",
        "tile_functor_cpu_bfloat16.cc",
        "tile_functor_cpu_bool.cc",
        "tile_functor_cpu_complex128.cc",
        "tile_functor_cpu_complex64.cc",
        "tile_functor_cpu_double.cc",
        "tile_functor_cpu_float.cc",
        "tile_functor_cpu_half.cc",
        "tile_functor_cpu_int16.cc",
        "tile_functor_cpu_int32.cc",
        "tile_functor_cpu_int64.cc",
        "tile_functor_cpu_int8.cc",
        "tile_functor_cpu_tstring.cc",
        "tile_functor_cpu_uint32.cc",
        "tile_functor_cpu_uint64.cc",
        "tile_functor_cpu_uint8.cc",
        "tile_functor_cpu_variant.cc",
        "tile_ops.cc",
        "tile_ops_cpu_impl_1.cc",
        "tile_ops_cpu_impl_2.cc",
        "tile_ops_cpu_impl_3.cc",
        "tile_ops_cpu_impl_4.cc",
        "tile_ops_cpu_impl_5.cc",
        "tile_ops_cpu_impl_6.cc",
        "tile_ops_cpu_impl_7.cc",
        "topk_op.cc",
        "training_op_helpers.cc",
        "training_ops.cc",
        "transpose_functor_cpu.cc",
        "transpose_op.cc",
        "unicode_ops.cc",
        "unique_op.cc",
        "unsorted_segment_join_op.cc",
        "where_op.cc",
        "whole_file_read_ops.cc",
        "xent_op.cc",
        ":portable_extended_ops_headers",
    ] + [
        "//tensorflow/core/kernels/data:portable_all_op_kernels",
        "//tensorflow/core/kernels/image:adjust_contrast_op.cc",
        "//tensorflow/core/kernels/image:adjust_hue_op.cc",
        "//tensorflow/core/kernels/image:adjust_saturation_op.cc",
        "//tensorflow/core/kernels/image:extract_image_patches_op.cc",
        "//tensorflow/core/kernels/image:image_ops.cc",
        "//tensorflow/core/kernels/image:mirror_pad_op.cc",
        "//tensorflow/core/kernels/image:mirror_pad_op_cpu_impl_1.cc",
        "//tensorflow/core/kernels/image:mirror_pad_op_cpu_impl_2.cc",
        "//tensorflow/core/kernels/image:mirror_pad_op_cpu_impl_3.cc",
        "//tensorflow/core/kernels/image:mirror_pad_op_cpu_impl_4.cc",
        "//tensorflow/core/kernels/image:mirror_pad_op_cpu_impl_5.cc",
        "//tensorflow/core/kernels/image:resize_bicubic_op.cc",
        "//tensorflow/core/kernels/image:resize_bilinear_op.cc",
        "//tensorflow/core/kernels/image:resize_nearest_neighbor_op.cc",
        "//tensorflow/core/kernels/image:sample_distorted_bounding_box_op.cc",
        "//tensorflow/core/kernels/linalg:cholesky_op.cc",
        "//tensorflow/core/kernels/linalg:determinant_op.cc",
        "//tensorflow/core/kernels/linalg:linalg_ops_common.cc",
        "//tensorflow/core/kernels/linalg:matrix_band_part_op.cc",
        "//tensorflow/core/kernels/linalg:matrix_diag_op.cc",
        "//tensorflow/core/kernels/linalg:matrix_inverse_op.cc",
        "//tensorflow/core/kernels/linalg:matrix_set_diag_op.cc",
        "//tensorflow/core/kernels/linalg:matrix_triangular_solve_op_complex.cc",
        "//tensorflow/core/kernels/linalg:matrix_triangular_solve_op_real.cc",
        "//tensorflow/core/kernels/linalg:qr_op_complex128.cc",
        "//tensorflow/core/kernels/linalg:qr_op_complex64.cc",
        "//tensorflow/core/kernels/linalg:qr_op_double.cc",
        "//tensorflow/core/kernels/linalg:qr_op_float.cc",
        "//tensorflow/core/kernels/linalg:qr_op_half.cc",
    ],
)

filegroup(
    name = "portable_quantized_ops",
    srcs = [
        "dequantize_op.cc",
        "meta_support.cc",
        "meta_support.h",
        "quantization_utils.cc",
        "quantization_utils.h",
        "quantize_down_and_shrink_range.cc",
        "quantize_op.cc",
        "quantized_activation_ops.cc",
        "quantized_add_op.cc",
        "quantized_batch_norm_op.cc",
        "quantized_bias_add_op.cc",
        "quantized_concat_op.cc",
        "quantized_conv_ops.cc",
        "quantized_instance_norm.cc",
        "quantized_matmul_op.cc",
        "quantized_mul_op.cc",
        "quantized_pooling_ops.cc",
        "quantized_reshape_op.cc",
        "quantized_resize_bilinear_op.cc",
        "reference_gemm.h",
        "requantization_range_op.cc",
        "requantize.cc",
        "reshape_op.h",
    ],
    visibility = ["//visibility:public"],
)

ANDROID_TEXTUAL_HDRS = [
    "//tensorflow/core/util:bad_indices_policy.h",
    "@local_xla//xla/tsl/framework/convolution:eigen_convolution_helpers.h",
    "@local_xla//xla/tsl/framework/convolution:eigen_spatial_convolutions-inl.h",
    "gather_nd_op_cpu_impl.h",
    "gemm_functors.h",
    "scatter_nd_op_cpu_impl.h",
    "slice_op_cpu_impl.h",
    "strided_slice_op_impl.h",
    "tile_ops_cpu_impl.h",
]

# A file group which contains nearly all available operators which
# may work on mobile environment. This is intended to be used with selective
# registration.
filegroup(
    name = "portable_all_ops",
    srcs = [
        "//tensorflow/c/kernels:portable_all_op_kernels",
        "//tensorflow/core/kernels/data:portable_all_op_kernels",
        "//tensorflow/core/kernels/image:portable_all_op_kernels",
        "//tensorflow/core/kernels/linalg:portable_all_op_kernels",
    ] + glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "*test.cc",
            "*test_util*",
            "*testutil*",
            "*testlib*",
            "*main.cc",
            "*_gpu*",
            "*_3d*",
            "*.cu.*",
            # Helper files for tests
            "eigen_benchmark.h",
            # Ops already in android_srcs
            "pooling_ops_common.cc",
            # Ops which we are currently excluding because they are likely
            # not used on Android. Those ops also do not compile if included,
            # unless we add the additional deps they need.
            "tf_record_reader_op.*",
            "cudnn_rnn_ops.*",
            "lmdb_reader_op.*",
            "string_to_hash_bucket_op.*",
            "sdca_ops.*",
            "sdca_internal.*",
            "sparse_cross_op.*",
            "text_line_reader_op.*",
            "summary_image_op.*",
            "identity_reader_op.*",
            "fixed_length_record_reader_op.*",
            "sample_distorted_bounding_box_op.*",
            "ctc_loss_op.*",
            "summary_interface.*",
            "summary_kernels.*",
            "spectrogram_convert_test_data.cc",
            "decode_proto_op.cc",
            "encode_proto_op.cc",
            "sobol_op.cc",
            # Excluded due to experimental status:
            "debug_ops.*",
            "mutex_ops.*",
            "batch_kernels.*",
            "string_lower_op.cc",  # Requires ICU for unicode.
            "string_upper_op.cc",  # Requires ICU for unicode.
            "unicode_ops.cc",
            "unicode_script_op.cc",
            # Ops that are inherently incompatible with Android (e.g. tied to x86 platform).
            "nextafter_op.cc",
            "initializable_lookup_table.*",
            "lookup_util.*",
            # Requires CUDA.
            "matmul_util.*",
        ] + ANDROID_TEXTUAL_HDRS,
    ) + [
        # Referenced by stateful_random_ops.cc but excluded with the *gpu*
        # rule above. Seems to have only have worked before because of
        # hdrs_check loose.
        "stateful_random_ops_cpu_gpu.h",
        # Allows conv_3d ops for android but excluded from *_3d* rule above.
        "conv_3d.h",
        "conv_ops_3d.cc",
        "conv_ops_gpu.h",
        # Transitive dependencies of the ops with attributes "bad_indices_policy".
        "//tensorflow/core/util:bad_indices_policy.cc",
        "//tensorflow/core/util:bad_indices_policy.h",
    ],
    visibility = ["//visibility:public"],
)

alias(
    name = "android_all_ops",
    actual = ":portable_all_ops",
)

filegroup(
    name = "portable_all_ops_textual_hdrs",
    srcs = ANDROID_TEXTUAL_HDRS + [
        "//tensorflow/core/kernels/image:portable_all_ops_textual_hdrs",
        "//tensorflow/core/util:image_resizer_state.h",
    ],
    visibility = ["//visibility:public"],
)

alias(
    name = "android_all_ops_textual_hdrs",
    actual = "portable_all_ops_textual_hdrs",
)

cc_library(
    name = "portable_tensorflow_kernels",
    srcs = if_mobile([
        "//tensorflow/core/kernels:portable_core_ops",
        "//tensorflow/core/kernels:portable_extended_ops",
    ]),
    copts = tf_copts() + tf_opts_nortti_if_lite_protos(),
    defines = ["EIGEN_NEON_GEBP_NR=4"],
    features = ["-layering_check"],
    linkopts = if_android(["-ldl"]),
    tags = [
        "manual",
        "notap",
    ],
    # These headers are not self-contained, so should be included in textual_hdrs only.
    textual_hdrs = ANDROID_TEXTUAL_HDRS,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:portable_jpeg_internal",
        "//tensorflow/core:portable_tensorflow_lib_lite",
        "//tensorflow/core/example:example_protos_cc",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "//tensorflow/core/framework:dataset_proto_cc",
        "//tensorflow/core/framework:full_type_proto_cc",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:model_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:reader_base_proto_cc",
        "//tensorflow/core/framework:summary_proto_cc",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/core/framework:tensor_shape_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/lib/png:png_io",
        "//tensorflow/core/platform:strong_hash",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/protobuf:autotuning_proto_cc",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/util/quantization:uniform_quant_ops_attr_proto_cc",
        "//third_party/fft2d:fft2d_headers",
        "//third_party/icu/data:conversion_data",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@ducc//:fft_wrapper",
        "@eigen_archive//:eigen3",
        "@fft2d",
        "@gemmlowp",
        "@icu//:common",
        "@local_xla//xla/tsl/framework/fixedpoint",
    ],
    alwayslink = 1,
)

build_test(
    name = "portable_tensorflow_kernels_build_test",
    targets = [":portable_tensorflow_kernels"],
)

cc_library(
    name = "android_whole_file_read_ops",
    srcs = if_android(["whole_file_read_ops.cc"]),
    copts = tf_copts(),
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:portable_tensorflow_lib_lite",
        "//tensorflow/core/framework:reader_base_proto_cc",
    ],
    alwayslink = 1,
)

#   Quantization-specific OpKernels

tf_kernel_library(
    name = "quantized_ops",
    srcs = [
        "dequantize_op.cc",
        "quantize_down_and_shrink_range.cc",
        "quantize_op.cc",
        "quantized_activation_ops.cc",
        "quantized_add_op.cc",
        "quantized_batch_norm_op.cc",
        "quantized_bias_add_op.cc",
        "quantized_concat_op.cc",
        "quantized_conv_ops.cc",
        "quantized_instance_norm.cc",
        "quantized_matmul_op.cc",
        "quantized_mul_op.cc",
        "quantized_pooling_ops.cc",
        "quantized_reshape_op.cc",
        "quantized_resize_bilinear_op.cc",
        "requantization_range_op.cc",
        "requantize.cc",
        "reshape_op.h",
    ],
    hdrs = ["reference_gemm.h"],
    features = ["-layering_check"],
    deps = [
        ":concat_lib_hdrs",
        ":conv_ops",
        ":cwise_op",
        ":eigen_helpers",
        ":meta_support",
        ":ops_util",
        ":pooling_ops",
        ":quantization_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core/util:determinism_for_kernels",
        "//tensorflow/core/util:image_resizer_state",
        "@eigen_archive//:eigen3",
        "@gemmlowp",
    ],
)

tf_cc_test(
    name = "requantization_range_op_test",
    size = "small",
    srcs = ["requantization_range_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantize_down_and_shrink_range_op_test",
    size = "small",
    srcs = ["quantize_down_and_shrink_range_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "requantize_op_test",
    size = "small",
    srcs = ["requantize_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantization_utils_test",
    srcs = ["quantization_utils_test.cc"],
    deps = [
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "@eigen_archive//:eigen3",
    ],
)

# Android-only test for quantization utilities.
tf_cc_binary(
    name = "quantization_utils_test_android_only",
    testonly = 1,
    srcs = ["quantization_utils_test.cc"],
    copts = tf_copts(),
    features = ["-layering_check"],
    linkopts = select({
        "//tensorflow:android": [
            "-lm",
            "-llog",
            "-pie",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
    ] + select({
        "//tensorflow:android": [
            ":portable_tensorflow_kernels",
            "//tensorflow/core:portable_tensorflow_lib",
            "//tensorflow/core:portable_tensorflow_test_lib",
        ],
        "//conditions:default": [
            ":quantized_ops",
            "//tensorflow/cc:cc_ops",
            "//tensorflow/cc:client_session",
            "//tensorflow/core:core_cpu_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:test",
            "//tensorflow/core/framework:tensor_testutil",
            "@eigen_archive//:eigen3",
        ],
    }),
)

tf_cc_test(
    name = "quantized_activation_ops_test",
    srcs = ["quantized_activation_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# Android-only test for quantized addition.
cc_binary(
    name = "quantized_add_op_test_android_only",
    testonly = 1,
    srcs = ["quantized_add_op_test.cc"],
    copts = tf_copts(),
    features = ["-layering_check"],
    linkopts = select({
        "//tensorflow:android": [
            "-lm",
            "-llog",
            "-pie",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
    ] + select({
        "//tensorflow:android": [
            ":portable_tensorflow_kernels",
            "//tensorflow/core:portable_tensorflow_lib",
            "//tensorflow/core:portable_tensorflow_test_lib",
        ],
        "//conditions:default": [
            ":ops_util",
            ":quantized_ops",
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:tensorflow",
            "//tensorflow/core:test",
            "//tensorflow/core/framework:tensor_testutil",
        ],
    }),
)

tf_cc_test(
    name = "quantized_add_op_test",
    size = "small",
    srcs = ["quantized_add_op_test.cc"],
    deps = [
        ":math",
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_resize_bilinear_op_test",
    size = "small",
    srcs = ["quantized_resize_bilinear_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:image_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
    ],
)

# Android-only test for quantized resize bilinear.
cc_binary(
    name = "quantized_resize_bilinear_op_test_android_only",
    testonly = 1,
    srcs = ["quantized_resize_bilinear_op_test.cc"],
    copts = tf_copts(),
    features = ["-layering_check"],
    linkopts = select({
        "//tensorflow:android": [
            "-lm",
            "-llog",
            "-pie",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
    ] + select({
        "//tensorflow:android": [
            ":portable_tensorflow_kernels",
            "//tensorflow/core:portable_tensorflow_lib",
            "//tensorflow/core:portable_tensorflow_test_lib",
        ],
        "//conditions:default": [
            ":ops_testutil",
            ":ops_util",
            ":quantized_ops",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:direct_session",
            "//tensorflow/core:framework",
            "//tensorflow/core:image_ops_op_lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:test",
            "//tensorflow/core:testlib",
        ],
    }),
)

tf_cc_test(
    name = "quantized_bias_add_op_test",
    size = "small",
    srcs = ["quantized_bias_add_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_conv_ops_test",
    size = "small",
    srcs = ["quantized_conv_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantize_op_test",
    size = "small",
    srcs = ["quantize_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_matmul_op_test",
    size = "small",
    srcs = ["quantized_matmul_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# Android-only test for quantized multiply.
cc_binary(
    name = "quantized_mul_op_test_android_only",
    testonly = 1,
    srcs = ["quantized_mul_op_test.cc"],
    features = ["-layering_check"],
    linkopts = select({
        "//tensorflow:android": [
            "-pie",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
    ] + select({
        "//tensorflow:android": [
            ":portable_tensorflow_kernels",
            "//tensorflow/core:portable_tensorflow_lib",
            "//tensorflow/core:portable_tensorflow_test_lib",
        ],
        "//conditions:default": [
            ":ops_util",
            ":quantized_ops",
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:test",
            "//tensorflow/core/framework:tensor_testutil",
        ],
    }),
)

tf_cc_test(
    name = "quantized_mul_op_test",
    size = "small",
    srcs = ["quantized_mul_op_test.cc"],
    deps = [
        ":math",
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_pooling_ops_test",
    size = "small",
    srcs = ["quantized_pooling_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_reshape_op_test",
    size = "small",
    srcs = ["quantized_reshape_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_concat_op_test",
    size = "small",
    srcs = ["quantized_concat_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_batch_norm_op_test",
    size = "small",
    srcs = ["quantized_batch_norm_op_test.cc"],
    deps = [
        ":batch_norm_op",
        ":ops_testutil",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@eigen_archive//:eigen3",
    ],
)

# Android-only test for quantized instance norm.
cc_binary(
    name = "quantized_instance_norm_test_android_only",
    testonly = 1,
    srcs = ["quantized_instance_norm_test.cc"],
    linkopts = select({
        "//tensorflow:android": [
            "-pie",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
    ] + select({
        "//tensorflow:android": [
            ":portable_tensorflow_kernels",
            "//tensorflow/core:portable_tensorflow_lib",
            "//tensorflow/core:portable_tensorflow_test_lib",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core/framework:tensor_testutil",
        ],
    }),
)

tf_cc_test(
    name = "quantized_instance_norm_test",
    size = "small",
    srcs = ["quantized_instance_norm_test.cc"],
    tags = ["no_mac_arm64"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core:framework",
        "@gemmlowp",
    ],
)

tf_cc_test(
    name = "bias_op_test",
    size = "small",
    srcs = ["bias_op_test.cc"],
    deps = [
        ":bias_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# NOTE(lespeholt): This rule is deprecated, please use:
# tensorflow/core/util/batch_util.h
cc_library(
    name = "batch_util",
    hdrs = ["batch_util.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(name = "boosted_trees_ops")

tf_kernel_library(
    name = "data_service_ops",
    deps = [
        "//tensorflow/core/kernels/data/experimental:data_service_kernels",
    ],
)

tf_kernel_library(
    name = "dataset_ops",
    deps = [
        "//tensorflow/core/kernels/data",
    ],
)

cc_library(
    name = "summary_interface",
    hdrs = ["summary_interface.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "summary_kernels",
    srcs = ["summary_kernels.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/db:sqlite",
        "//tensorflow/core/summary:schema",
        "//tensorflow/core/summary:summary_db_writer",
        "//tensorflow/core/summary:summary_file_writer",
    ],
)

tf_kernel_library(
    name = "decode_proto_op",
    srcs = [
        "decode_proto_op.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util/proto:decode",
        "//tensorflow/core/util/proto:descriptors",
        "//tensorflow/core/util/proto:proto_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "encode_proto_op",
    srcs = ["encode_proto_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util/proto:descriptors",
        "//tensorflow/core/util/proto:proto_utils",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "unicode_script_op",
    srcs = ["unicode_script_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "@icu//:common",
    ],
)

tf_kernel_library(
    name = "sync_ops",
    features = ["-layering_check"],
    prefix = "sync_ops",
    deps = [
        "//tensorflow/core:framework",
    ],
)

# Library to link with when compiling the cwise_op kernels directly,
# e.g. for selective registration.
cc_library(
    name = "cwise_lib",
    hdrs = [
        "cwise_ops.h",
        "cwise_ops_common.h",
        "cwise_ops_gpu_common.cu.h",
        "cwise_ops_gpu_gradients.cu.h",
        "cwise_ops_gradients.h",
        "fill_functor.h",
    ],
    deps = [
        ":meta_support",
        ":quantization_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
        "@eigen_archive//:eigen3",
        "@gemmlowp",
    ],
)

# Header-only version of cwise_lib for clients that want to use the cwise_ops
# functionality in their own custom ops.
cc_header_only_library(
    name = "cwise_lib_hdrs",
    deps = [
        ":cwise_lib",
    ],
)

# Library to link with when compiling the quantize and dequantize kernels directly,
# e.g. for selective registration.
cc_header_only_library(
    name = "quantize_and_dequantize_op_hdrs",
    deps = [
        ":quantize_and_dequantize_op",
    ],
)

cc_library(
    name = "kernel_platform_strings",
    srcs = ["kernel_platform_strings.h"],
    deps = [
        "//tensorflow/core/platform:platform_strings",
    ],
    alwayslink = 1,
)

# For a more maintainable build this target should not exist and the headers
# should  be split into the existing cc_library targets, but this change was
# automatically  done so that we can remove long standing issues and complexity
# in the build system. It's up to the OWNERS of this package to get rid of it or
# not. The use of the textual_hdrs attribute is discouraged, use hdrs instead.
# Here it is used to avoid header parsing errors in packages where the feature
# parse_headers was enabled since loose headers were not being parsed. See
# go/loose-lsc-one-target-approach for more details.
cc_library(
    name = "loose_headers",
    tags = ["avoid_dep"],
    textual_hdrs = glob(["*.h"]),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
    ],
)

tf_kernel_library(
    name = "stochastic_cast_op",
    features = ["-layering_check"],
    prefix = "stochastic_cast_op",
    deps = [
        ":stateless_random_ops_v2_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "stochastic_cast_op_test",
    timeout = "moderate",
    srcs = ["stochastic_cast_op_test.cc"],
    features = ["-layering_check"],
    shard_count = 48,
    deps = [
        ":cwise_lib",
        ":ops_testutil",
        ":ops_util",
        ":stochastic_cast_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:fake_input",
        "//tensorflow/core/framework:types_proto_cc",
        "@eigen_archive//:eigen3",
    ],
)

# Shared object that links all the kernels TF needs.
tf_cc_shared_library(
    name = "libtfkernel_all_kernels.so",
    static_deps = [
        # TODO(rostam): Remove Copybara transform.
        # copybara:comment_begin(oss only)
        "@bazel_tools//:__subpackages__",
        "@boringssl//:__subpackages__",
        "@com_github_cares_cares//:__subpackages__",
        "@com_github_googlecloudplatform_tensorflow_gcp_tools//:__subpackages__",
        "@com_github_grpc_grpc//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@com_google_googleapis//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@com_googlesource_code_re2//:__subpackages__",
        "@compute_library//:__subpackages__",
        "@curl//:__subpackages__",
        "@eigen_archive//:__subpackages__",
        "@farmhash_archive//:__subpackages__",
        "@farmhash_gpu_archive//:__subpackages__",
        "@fft2d//:__subpackages__",
        "@gemmlowp//:__subpackages__",
        "@gif//:__subpackages__",
        "@highwayhash//:__subpackages__",
        "@hwloc//:__subpackages__",
        "@icu//:__subpackages__",
        "@jsoncpp_git//:__subpackages__",
        "@libjpeg_turbo//:__subpackages__",
        "@llvm_openmp//:__subpackages__",
        "@llvm-project//:__subpackages__",
        "@llvm_terminfo//:__subpackages__",
        "@llvm_zlib//:__subpackages__",
        "@local_config_cuda//:__subpackages__",
        "@local_config_git//:__subpackages__",
        "@local_config_nccl//:__subpackages__",
        "@local_config_rocm//:__subpackages__",
        "@local_config_tensorrt//:__subpackages__",
        "@local_execution_config_platform//:__subpackages__",
        "@mkl_dnn_acl_compatible//:__subpackages__",
        "@net_zstd//:__subpackages__",
        "@onednn//:__subpackages__",
        "@org_sqlite//:__subpackages__",
        "@platforms//:__subpackages__",
        "@png//:__subpackages__",
        "@snappy//:__subpackages__",
        "@stablehlo//:__subpackages__",
        "//:__subpackages__",
        "@upb//:__subpackages__",
        "@zlib//:__subpackages__",
        "@local_tsl//tsl:__subpackages__",
        "@local_xla//xla:__subpackages__",
        # copybara:comment_end
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kernel_platform_strings",
        "//tensorflow/core:all_kernels_impl",
    ],
)

# Manually curated set of tests that are useful for building and testing against
# platforms and architecures that don't support CUDA.
# TODO(b/153737462): Automatically filter tests to create the appropriate
# portable test list.
test_suite(
    name = "portable_kernel_tests",
    tags = [
        "manual",  # Avoid redundancy when using wildcard test patterns.
    ],
    tests = [
        ":batch_norm_op_test",
        ":broadcast_to_op_test_cpu",
        ":cast_op_test_cpu",
        ":concat_op_test",
        ":control_flow_ops_test",
        ":cwise_ops_test_cpu",
        ":deep_conv2d_test",
        ":dequantize_op_test",
        ":diag_op_test_cpu",
        ":eigen_activations_test",
        ":eigen_pooling_test",
        ":gather_nd_op_test_cpu",
        ":matmul_op_test_cpu",
        ":mfcc_test",
        ":multinomial_op_test_cpu",
        ":nn_ops_test_cpu",
        ":quantization_utils_test",
        ":quantize_and_dequantize_op_test_cpu",
        ":quantize_op_test",
        ":quantized_activation_ops_test",
        ":quantized_batch_norm_op_test",
        ":quantized_conv_ops_test",
        ":quantized_matmul_op_test",
        ":quantized_pooling_ops_test",
        ":random_binomial_op_test_cpu",
        ":random_op_test_cpu",
        ":random_poisson_op_test_cpu",
        ":reduction_ops_test_cpu",
        ":requantization_range_op_test",
        ":scatter_op_test",
        ":segment_reduction_ops_test",
        ":slice_op_test",
        ":spectrogram_test",
        ":split_op_test",
        ":split_v_op_test_cpu",
        ":strided_slice_op_test",
        ":unique_op_test",
        ":variable_ops_test",
        "//tensorflow/core/kernels/image:crop_and_resize_op_test",
        "//tensorflow/core/kernels/image:non_max_suppression_op_test",
        "//tensorflow/core/kernels/image:resize_ops_test_cpu",
    ],
)

exports_files(
    glob([
        "cwise_op*.cc",
        "cwise_op_gpu*.cu.cc",
    ]) + [
        "dequantize_op.cc",
        "ops_testutil.h",
        "quantize_and_dequantize_op.cc",
        "quantize_op.cc",
        "sparse_cross_op.cc",
        "sparse_fill_empty_rows_op.cc",
        "sparse_reshape_op.cc",
        "unary_ops_composition.cc",
    ],
)

tf_kernel_library(
    name = "sobol_op",
    srcs = [
        "sobol_op.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/numeric:bits",
        "@eigen_archive//:eigen3",
        "@sobol_data",
    ],
)

# ---- temporary forwarding declaration for libraries in linalg
# TODO(b/160344057): Remove after updating dependencies.
tf_kernel_library(
    name = "matrix_inverse_op",
    deps = ["//tensorflow/core/kernels/linalg:matrix_inverse_op"],
)

tf_kernel_library(
    name = "einsum_op",
    deps = ["//tensorflow/core/kernels/linalg:einsum_op"],
)

tf_kernel_library(
    name = "isotonic_regression_op",
    srcs = [
        "isotonic_regression_op.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "isotonic_regression_op_test",
    size = "small",
    srcs = ["isotonic_regression_op_test.cc"],
    deps = [
        ":isotonic_regression_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "filesystem_ops",
    srcs = ["filesystem_ops.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)
