load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm",
)
load(
    "//tensorflow:tensorflow.bzl",
    "check_deps",
    "tf_cc_test",
    "tf_cc_test_mkl",
    "tf_cc_tests",
    "tf_copts",
    "tf_cuda_library",
    "tf_cuda_only_cc_test",
    "tf_kernel_library",
    "tf_mkl_kernel_library",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup", "get_compatible_with_portable", "tf_version_info_genrule")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

default_package_visibility = [
    "//tensorflow/core:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_package_visibility,
    features = [
        "-parse_headers",
    ],
    licenses = ["notice"],
)

# List of exported source files.
exports_files(
    srcs = [
        "bad_indices_policy.cc",
        "bad_indices_policy.h",
        "event.proto",
        "example_proto_fast_parsing_test.proto",
        "image_resizer_state.h",
        "memmapped_file_system.proto",
        "saved_tensor_slice.proto",
    ],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/core:__pkg__",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "overflow.h",
        "padding.cc",
        "padding.h",
        "tensor_format.cc",
        "tensor_format.h",
        "@local_xla//xla/tsl/util:mobile_srcs_no_runtime",
    ],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "activation_mode.h",
        "batch_util.cc",
        "batch_util.h",
        "bcast.cc",
        "bcast.h",
        "command_line_flags.h",
        "debug_data_dumper.cc",
        "debug_data_dumper.h",
        "determinism.h",
        "device_name_utils.h",
        "dump_graph.cc",
        "dump_graph.h",
        "einsum_op_util.cc",
        "einsum_op_util.h",
        "env_var.h",
        "equal_graph_def.cc",
        "equal_graph_def.h",
        "example_proto_fast_parsing.cc",
        "example_proto_fast_parsing.h",
        "example_proto_helper.cc",
        "example_proto_helper.h",
        "guarded_philox_random.cc",
        "guarded_philox_random.h",
        "managed_stack_trace.h",
        "matmul_autotune.cc",
        "matmul_autotune.h",
        "matmul_bcast.h",
        "mirror_pad_mode.cc",
        "mirror_pad_mode.h",
        "port.cc",
        "port.h",
        "presized_cuckoo_map.h",
        "ragged_to_dense_util.cc",
        "ragged_to_dense_util.h",
        "ragged_to_dense_util_common.h",
        "reffed_status_callback.h",
        "saved_tensor_slice_util.cc",
        "saved_tensor_slice_util.h",
        "stat_summarizer.cc",
        "stat_summarizer.h",
        "strided_slice_op.cc",
        "strided_slice_op.h",
        "tensor_ops_util.h",
        "tensor_slice_reader.cc",
        "tensor_slice_reader.h",
        "tensor_slice_reader_cache.cc",
        "tensor_slice_reader_cache.h",
        "tensor_slice_set.cc",
        "tensor_slice_set.h",
        "tensor_slice_util.h",
        "tensor_slice_writer.cc",
        "tensor_slice_writer.h",
        "use_cudnn.h",
        "util.cc",
        "util.h",
        "work_sharder.cc",
        "work_sharder.h",
        "//tensorflow/core/config:mobile_srcs_only_runtime",
        "//tensorflow/core/util/quantization:mobile_srcs_only_runtime",
        "@local_xla//xla/tsl/util:mobile_srcs_only_runtime",
    ],
)

filegroup(
    name = "framework_internal_private_hdrs",
    srcs = [
        "activation_mode.h",
        "batch_util.h",
        "bcast.h",
        "command_line_flags.h",
        "debug_data_dumper.h",
        "debug_events_writer.h",
        "device_name_utils.h",
        "dump_graph.h",
        "einsum_op_util.h",
        "env_var.h",
        "equal_graph_def.h",
        "events_writer.h",
        "example_proto_fast_parsing.h",
        "example_proto_helper.h",
        "exec_on_stall.h",
        "gpu_cuda_alias.h",
        "gpu_device_functions.h",
        "gpu_kernel_helper.h",
        "gpu_launch_config.h",
        "guarded_philox_random.h",
        "matmul_autotune.h",
        "matmul_bcast.h",
        "mirror_pad_mode.h",
        "mkl_heuristics.h",
        "mkl_util.h",
        "onednn_env_vars.h",
        "overflow.h",
        "padding.h",
        "permutation_input_iterator.h",
        "permutation_output_iterator.h",
        "presized_cuckoo_map.h",
        "reffed_status_callback.h",
        "saved_tensor_slice_util.h",
        "stat_summarizer.h",
        "stat_summarizer_options.h",
        "stats_calculator.h",
        "stream_executor_util.h",
        "strided_slice_op.h",
        "tensor_format.h",
        "tensor_ops_util.h",
        "tensor_slice_reader.h",
        "tensor_slice_reader_cache.h",
        "tensor_slice_set.h",
        "tensor_slice_util.h",
        "tensor_slice_writer.h",
        "transform_output_iterator.h",
        "use_cudnn.h",
        "util.h",
        "work_sharder.h",
        "xla_config_registry.h",
        "@local_xla//xla/tsl/util:framework_internal_private_hdrs",
    ],
)

filegroup(
    name = "framework_internal_public_hdrs",
    srcs = [
        "command_line_flags.h",
        "equal_graph_def.h",
        "presized_cuckoo_map.h",
        "tensor_slice_set.h",
        "tensor_slice_util.h",
        "xla_config_registry.h",
    ],
)

filegroup(
    name = "framework_internal_impl_srcs",
    srcs = [
        "activation_mode.cc",
        "batch_util.cc",
        "bcast.cc",
        "debug_data_dumper.cc",
        "debug_events_writer.cc",
        "dump_graph.cc",
        "equal_graph_def.cc",
        "events_writer.cc",
        "example_proto_fast_parsing.cc",
        "example_proto_helper.cc",
        "guarded_philox_random.cc",
        "matmul_autotune.cc",
        "mirror_pad_mode.cc",
        "saved_tensor_slice_util.cc",
        "stat_summarizer.cc",
        "strided_slice_op.cc",
        "tensor_slice_reader.cc",
        "tensor_slice_reader_cache.cc",
        "tensor_slice_set.cc",
        "tensor_slice_writer.cc",
        "util.cc",
        "work_sharder.cc",
        "xla_config_registry.cc",
        "@local_xla//xla/tsl/util:framework_internal_impl_srcs",
    ],
)

filegroup(
    name = "lib_internal_public_hdrs",
    srcs = [
        "env_var.h",
        "use_cudnn.h",
        "@local_xla//xla/tsl/util:lib_internal_public_hdrs",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "lib_hdrs",
    srcs = [
        "gpu_cuda_alias.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "memmapped_file_system_hdrs",
    srcs = [
        "memmapped_file_system.h",
        "memmapped_file_system_writer.h",
    ],
)

filegroup(
    name = "memmapped_file_system_srcs",
    srcs = [
        "memmapped_file_system.cc",
        "memmapped_file_system_writer.cc",
    ],
)

filegroup(
    name = "port_hdrs",
    srcs = [
        "port.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/util:__pkg__",
    ],
)

filegroup(
    name = "test_hdrs",
    testonly = 1,
    srcs = [
        "reporter.h",
        "@local_xla//xla/tsl/util:test_hdrs",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mkl_util_hdrs",
    srcs = [
        "mkl_heuristics.h",
        "mkl_util.h",
        "onednn_env_vars.h",
        "@local_xla//xla/tsl/util:onednn_util_hdrs",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mkl_util_test_srcs",
    testonly = 1,
    srcs = [
        "mkl_util_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "android_test_hdrs",
    testonly = 1,
    srcs = [
        "reporter.h",
        "@local_xla//xla/tsl/util:android_test_hdrs",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "android_test_srcs",
    testonly = 1,
    srcs = [
        ":android_test_hdrs",
        "@local_xla//xla/tsl/util:android_test_srcs",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "framework_srcs",
    srcs = [
        "activation_mode.h",
        "batch_util.h",
        "bcast.h",
        "debug_data_dumper.h",
        "debug_events_writer.h",
        "device_name_utils.h",
        "dump_graph.h",
        "einsum_op_util.h",
        "events_writer.h",
        "example_proto_fast_parsing.h",
        "example_proto_helper.h",
        "gpu_kernel_helper.h",
        "guarded_philox_random.h",
        "managed_stack_trace.h",
        "matmul_autotune.h",
        "matmul_bcast.h",
        "mirror_pad_mode.h",
        "padding.h",
        "port.h",
        "reffed_status_callback.h",
        "saved_tensor_slice_util.h",
        "stat_summarizer.h",
        "stat_summarizer_options.h",
        "stream_executor_util.h",
        "strided_slice_op.h",
        "tensor_format.h",
        "tensor_ops_util.h",
        "tensor_slice_reader.h",
        "tensor_slice_reader_cache.h",
        "tensor_slice_writer.h",
        "use_cudnn.h",
        "util.h",
        "work_sharder.h",
        "@local_xla//xla/tsl/util:framework_srcs",
    ],
)

# Version info generation needs to be generated in the same package where it
# is written.
tf_version_info_genrule(
    name = "version_info_gen",
    out = "version_info.h",
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "version_info",
    hdrs = ["version_info.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fake_clock_env",
    testonly = 1,
    srcs = ["fake_clock_env.cc"],
    hdrs = ["fake_clock_env.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_test(
    name = "fake_clock_env_test",
    srcs = [
        "fake_clock_env_test.cc",
    ],
    deps = [
        ":fake_clock_env",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:env",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "stats_calculator_portable",
    hdrs = [
        "stat_summarizer_options.h",
        "stats_calculator.h",
    ],
    copts = tf_copts(),
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "@local_xla//xla/tsl/util:stats_calculator_portable",
    ],
)

cc_library(
    name = "stat_summarizer",
    hdrs = [
        "stat_summarizer.h",
    ],
    visibility = [
        "//tensorflow/core:__subpackages__",
        "//tensorflow/python/util:__pkg__",
    ],
    deps = [
        ":stats_calculator_portable",
        "//tensorflow/core:framework",
    ],
)

tf_mkl_kernel_library(
    name = "onednn_env_vars",
    srcs = ["onednn_env_vars.cc"],
    hdrs = ["onednn_env_vars.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/kernels:ops_util",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "ragged_to_dense_util_common",
    hdrs = [
        "ragged_to_dense_util_common.h",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "ragged_to_dense_util",
    srcs = [
        "ragged_to_dense_util.cc",
    ],
    hdrs = [
        "ragged_to_dense_util.h",
    ],
    deps = [
        ":ragged_to_dense_util_common",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "ragged_to_dense_util_test",
    srcs = [
        "ragged_to_dense_util_test.cc",
    ],
    deps = [
        ":ragged_to_dense_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "reporter",
    hdrs = ["reporter.h"],
    visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    deps = [
        ":test_log_proto_cc",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:env_impl",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:types",
        "@local_xla//xla/tsl/util:reporter",
    ],
)

cc_library(
    name = "port",
    srcs = ["port.cc"],
    hdrs = ["port.h"],
    copts = tf_copts(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/util:__pkg__",
    ],
    deps = [
        ":env_var",
        "//tensorflow/core/platform:platform_port",
        "@com_google_absl//absl/base",
    ],
    alwayslink = 1,
)

cc_library(
    name = "managed_stack_trace",
    hdrs = ["managed_stack_trace.h"],
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/c/eager:__pkg__",
        "//tensorflow/compiler/mlir/tensorflow:__pkg__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/common_runtime/eager:__pkg__",
        "//tensorflow/core/framework:__pkg__",
        "//tensorflow/core/platform:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/eager:__pkg__",
        "//tensorflow/python/util:__pkg__",
    ],
    deps = [
        "//tensorflow/core/platform:stack_frame",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cuda_library(
    name = "gpu_cuda_alias",
    hdrs = ["gpu_cuda_alias.h"],
)

tf_cuda_library(
    name = "gpu_device_functions",
    hdrs = ["gpu_device_functions.h"],
    deps = [
        ":gpu_cuda_alias",
        "//tensorflow/core/platform:types",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "overflow",
    hdrs = ["overflow.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "exec_on_stall",
    hdrs = ["exec_on_stall.h"],
    deps = [
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:mutex",
    ],
)

cc_library(
    name = "tensor_format",
    srcs = ["tensor_format.cc"],
    hdrs = ["tensor_format.h"],
    deps = [
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/lib/gtl:inlined_vector",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        ":tensor_format",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
    ],
)

cc_library(
    name = "einsum_op_util",
    srcs = ["einsum_op_util.cc"],
    hdrs = ["einsum_op_util.h"],
    deps = [
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/gtl:inlined_vector",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "env_var",
    hdrs = ["env_var.h"],
    deps = [
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:numbers",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "@local_xla//xla/tsl/util:env_var",
    ],
)

cc_library(
    name = "incremental_barrier",
    srcs = ["incremental_barrier.cc"],
    hdrs = ["incremental_barrier.h"],
    visibility = default_package_visibility + ["//tensorflow_serving:__subpackages__"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

# Note: This rule should not be used as a dependency for kernels. Use the
# "determinism_for_kernels" rule below instead.
cc_library(
    name = "determinism",
    hdrs = ["determinism.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":env_var",
        "//tensorflow/core/platform:mutex",
        "@local_xla//xla/tsl/util:determinism",
    ],
    alwayslink = 1,
)

filegroup(
    name = "determinism_hdr",
    srcs = [
        "determinism.h",
        "@local_xla//xla/tsl/util:determinism_hdr",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "determinism_hdr_lib",
    hdrs = [":determinism_hdr"],
    compatible_with = get_compatible_with_portable(),
    # TODO(b/298501506): narrow this in a way that won't break TFRT
    visibility = ["//visibility:public"],
    deps = [
        "@local_xla//xla/tsl/util:determinism_hdr_lib",
    ],
)

# This alias should be used as a dependency for kernels which use determinism,
# as well any other rules which are in the same shared library as the kernels.
# This rule does not include the determinism.cc file for nonstatic builds. The
# reason is that for nonstatic builds, the shared object which contains the
# kernels (e.g. _pywrap_tensorflow_internal.so) must not contain the global
# variable in determinism.cc, since the global variable is already in
# libtensorflow_framework.so.
#
# To test that determinism.cc is not improperly included in the shared object
# which contains the kernels, you can run the "determinism_check_deps" rule
# below.
alias(
    name = "determinism_for_kernels",
    actual = if_static(":determinism", ":determinism_hdr_lib"),
    visibility = ["//tensorflow:__subpackages__"],
)

check_deps(
    name = "determinism_check_deps",
    disallowed_deps = if_static(
        [],
        otherwise = [":determinism"],
    ),
    deps = [
        "//tensorflow:tensorflow_cc",
    ],
)

tf_kernel_library(
    name = "cuda_solvers",
    srcs = ["cuda_solvers.cc"],
    hdrs = ["gpu_solvers.h"],
    compatible_with = [],
    features = ["-layering_check"],
    visibility = ["//tensorflow/core/kernels:friends"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@local_xla//xla/stream_executor/cuda:cublas_plugin",
        "@local_xla//xla/tsl/cuda:cusolver",
    ],
)

tf_kernel_library(
    name = "rocm_solvers",
    srcs = ["rocm_solvers.cc"],
    hdrs = ["gpu_solvers.h"],
    compatible_with = [],
    visibility = ["//tensorflow/core/kernels:friends"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@local_xla//xla/stream_executor/platform:dso_loader",
        "@local_xla//xla/stream_executor/rocm:hipsolver_wrapper",
        "@local_xla//xla/stream_executor/rocm:rocblas_plugin",
        "@local_xla//xla/stream_executor/rocm:rocblas_wrapper",
        "@local_xla//xla/stream_executor/rocm:rocm_executor",
        "@local_xla//xla/stream_executor/rocm:rocsolver_wrapper",
    ] + if_rocm([
        "@local_config_rocm//rocm:rocprim",
    ]),
)

tf_kernel_library(
    name = "cuda_sparse",
    srcs = if_cuda(["cuda_sparse.cc"]) + if_rocm(["rocm_sparse.cc"]),
    hdrs = [
        "cuda_sparse.h",
        "gpu_solvers.h",
    ],
    compatible_with = [],
    features = ["-layering_check"],
    deps = [
        ":cuda_solvers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ] + if_cuda([
        "@local_xla//xla/stream_executor/cuda:cuda_blas_utils",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor:dnn",
        "@local_xla//xla/tsl/cuda:cusparse",
        "@local_config_cuda//cuda:cub_headers",
    ]) + if_rocm([
        "@local_xla//xla/stream_executor/rocm:rocsolver_wrapper",
        "@local_xla//xla/stream_executor/rocm:hipsolver_wrapper",
        "@local_xla//xla/stream_executor/rocm:hipsparse_wrapper",
    ]),
)

cc_library(
    name = "image_resizer_state",
    hdrs = ["image_resizer_state.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:bounds_check",
        "@eigen_archive//:eigen3",
    ],
)

# Tests.
tf_cc_test(
    name = "overflow_test",
    size = "small",
    srcs = ["overflow_test.cc"],
    deps = [
        ":overflow",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "exec_on_stall_test",
    size = "small",
    srcs = ["exec_on_stall_test.cc"],
    deps = [
        ":exec_on_stall",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_only_cc_test(
    name = "gpu_kernel_helper_test",
    srcs = [
        "gpu_kernel_helper_test.cu.cc",
    ],
    features = ["-layering_check"],
    tags = [
        "no_cuda_asan",  # TODO(b/171342366): re-enable.
    ],
    deps = [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_tests(
    name = "higher_level_tests",
    size = "small",
    srcs = [
        "bcast_test.cc",
        "command_line_flags_test.cc",
        "debug_data_dumper_test.cc",
        "dump_graph_test.cc",
        "equal_graph_def_test.cc",
        "events_writer_test.cc",
        "example_proto_fast_parsing_test.cc",
        "example_proto_helper_test.cc",
        "matmul_bcast_test.cc",
        "memmapped_file_system_test.cc",
        "presized_cuckoo_map_test.cc",
        "reffed_status_callback_test.cc",
        "reporter_test.cc",
        "saved_tensor_slice_util_test.cc",
        "semver_test.cc",
        "stat_summarizer_test.cc",
        "strided_slice_op_test.cc",
        "tensor_format_test.cc",
        "tensor_slice_reader_test.cc",
        "tensor_slice_set_test.cc",
        "tensor_slice_util_test.cc",
        "tensor_slice_writer_test.cc",
        "work_sharder_test.cc",
    ],
    features = ["-layering_check"],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    visibility = [
        "//tensorflow/core:__pkg__",
    ],
    deps = [
        ":protos_test_cc",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/cc:while_loop",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/platform:status_matchers",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "debug_events_writer_test",
    size = "small",
    srcs = ["debug_events_writer_test.cc"],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "incremental_barrier_test",
    srcs = ["incremental_barrier_test.cc"],
    deps = [
        ":incremental_barrier",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test_mkl(
    name = "mkl_heuristics_test",
    size = "small",
    srcs = ["mkl_heuristics_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:graph",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

# Proto libraries.
tf_proto_library(
    name = "test_log_proto",
    srcs = ["test_log.proto"],
    make_default_target_header_only = True,
    protodeps = ["@local_tsl//tsl/protobuf:test_log_proto"],
    # Not to be used outside of tensorflow/core.
    visibility = ["//tensorflow/core:__pkg__"],
    exports = ["@local_tsl//tsl/protobuf:test_log_proto"],
)

tf_proto_library(
    name = "protos_test",
    srcs = ["example_proto_fast_parsing_test.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/core:protos_all"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "event_proto",
    srcs = ["event.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/core/framework:summary_proto",
        "//tensorflow/core/framework:resource_handle_proto",
        "//tensorflow/core/framework:tensor_proto",
        "//tensorflow/core/framework:tensor_shape_proto",
        "//tensorflow/core/framework:types_proto",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "saved_tensor_slice_proto",
    srcs = ["saved_tensor_slice.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/core/framework:resource_handle_proto",
        "//tensorflow/core/framework:tensor_proto",
        "//tensorflow/core/framework:tensor_shape_proto",
        "//tensorflow/core/framework:tensor_slice_proto",
        "//tensorflow/core/framework:types_proto",
        "//tensorflow/core/framework:versions_proto",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "memmapped_file_system_proto",
    srcs = ["memmapped_file_system.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "protos_all",
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":event_proto",
        ":saved_tensor_slice_proto",
        ":memmapped_file_system_proto",
        "//tensorflow/core/util/quantization:uniform_quant_ops_attr_proto",
    ],
    tags = [
        "alt_dep=//third_party/tensorflow/core:protos_all",
    ],
    visibility = [
        "//tensorflow/core:__subpackages__",
        "@local_tsl//tsl:__subpackages__",
    ],
)

cc_library(
    name = "bad_indices_policy",
    srcs = ["bad_indices_policy.cc"],
    hdrs = ["bad_indices_policy.h"],
    deps = [
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tf_cc_test(
    name = "bad_indices_policy_test",
    srcs = ["bad_indices_policy_test.cc"],
    deps = [
        ":bad_indices_policy",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
