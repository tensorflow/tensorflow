load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
)
load("//tensorflow:tensorflow.bzl", "tf_version_info_genrule")

package(
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Export all files for which we do not yet provide a dedicated build rule.
# This avoids breading all the rules in tensorflow/core/BUILD.
exports_files(
    srcs = [
        "activation_mode.cc",
        "activation_mode.h",
        "batch_util.cc",
        "batch_util.h",
        "bcast.cc",
        "bcast.h",
        "command_line_flags.cc",
        "command_line_flags.h",
        "debug_events_writer.cc",
        "debug_events_writer.h",
        "device_name_utils.cc",
        "device_name_utils.h",
        "dump_graph.cc",
        "dump_graph.h",
        "einsum_op_util.cc",
        "einsum_op_util.h",
        "env_var.cc",
        "env_var.h",
        "equal_graph_def.cc",
        "equal_graph_def.h",
        "events_writer.cc",
        "events_writer.h",
        "example_proto_fast_parsing.cc",
        "example_proto_fast_parsing.h",
        "example_proto_helper.cc",
        "example_proto_helper.h",
        "exec_on_stall.h",
        "gpu_cuda_alias.h",
        "gpu_device_functions.h",
        "gpu_kernel_helper.h",
        "gpu_kernel_helper_test.cu.cc",
        "gpu_launch_config.h",
        "guarded_philox_random.cc",
        "guarded_philox_random.h",
        "matmul_autotune.cc",
        "matmul_autotune.h",
        "matmul_bcast.cc",
        "matmul_bcast.h",
        "memmapped_file_system.cc",
        "memmapped_file_system.h",
        "memmapped_file_system_writer.cc",
        "memmapped_file_system_writer.h",
        "mirror_pad_mode.cc",
        "mirror_pad_mode.h",
        "mkl_util.h",
        "overflow.h",
        "padding.cc",
        "padding.h",
        "permutation_input_iterator.h",
        "permutation_output_iterator.h",
        "port.cc",
        "port.h",
        "presized_cuckoo_map.h",
        "ptr_util.h",
        "reffed_status_callback.h",
        "reporter.cc",
        "reporter.h",
        "saved_tensor_slice_util.cc",
        "saved_tensor_slice_util.h",
        "stat_summarizer.cc",
        "stat_summarizer.h",
        "stat_summarizer_options.h",
        "stats_calculator.cc",
        "stats_calculator.h",
        "stream_executor_util.h",
        "strided_slice_op.cc",
        "strided_slice_op.h",
        "tensor_format.cc",
        "tensor_format.h",
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
        "transform_output_iterator.h",
        "use_cudnn.cc",
        "use_cudnn.h",
        "util.cc",
        "util.h",
        "work_sharder.cc",
        "work_sharder.h",
        "xla_config_registry.cc",
        "xla_config_registry.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

# List of exported test source files that do not yet have local build rules.
exports_files(
    srcs = [
        "bcast_test.cc",
        "command_line_flags_test.cc",
        "debug_events_writer_test.cc",
        "device_name_utils_test.cc",
        "dump_graph_test.cc",
        "equal_graph_def_test.cc",
        "events_writer_test.cc",
        "example_proto_fast_parsing_test.cc",
        "example_proto_helper_test.cc",
        "exec_on_stall_test.cc",
        "matmul_bcast_test.cc",
        "memmapped_file_system_test.cc",
        "mkl_util_test.cc",
        "overflow_test.cc",
        "presized_cuckoo_map_test.cc",
        "reffed_status_callback_test.cc",
        "reporter_test.cc",
        "saved_tensor_slice_util_test.cc",
        "semver_test.cc",
        "stat_summarizer_test.cc",
        "stats_calculator_test.cc",
        "tensor_format_test.cc",
        "tensor_slice_reader_test.cc",
        "tensor_slice_set_test.cc",
        "tensor_slice_util_test.cc",
        "tensor_slice_writer_test.cc",
        "work_sharder_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

# List of exported proto source files.
exports_files(
    srcs = [
        "event.proto",
        "example_proto_fast_parsing_test.proto",
        "memmapped_file_system.proto",
        "saved_tensor_slice.proto",
        "test_log.proto",
    ],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/core:__pkg__",
    ],
)

# The following filegroups are needed since globbing across packages boundaries
# will just fail silently (see 3rd caveat at
# https://docs.bazel.build/versions/master/be/functions.html#glob).
# Files needed for core:framework_internal_impl.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "*_test.*",
            "debug_events_writer.*",
            "stats_calculator.*",
            "events_writer.*",
            "reporter.*",
        ],
    ),
)

filegroup(
    name = "lib_internal_impl_srcs",
    srcs = ["env_var.cc"],
)

filegroup(
    name = "framework_internal_private_hdrs",
    srcs = glob(
        [
            "*.h",
        ],
        exclude = [
            "reporter.h",
            "port.h",
            "memmapped_file_system.*",
            "memmapped_file_system_writer.*",
            "session_message.*",
        ],
    ),
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
    srcs = glob(
        [
            "*.cc",
        ],
        exclude = [
            "*test*",
            "env_var.cc",
            "memmapped_file_system.*",
            "memmapped_file_system_writer.*",
            "port.cc",
            "reporter.cc",
            "stats_calculator.cc",
            "version_info.cc",
        ],
    ),
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
    ],
)

# Version info generation needs to be generated in the same package where it
# is written.
tf_version_info_genrule(
    name = "version_info_gen",
    out = "version_info.cc",
)

cc_library(
    name = "stats_calculator_portable",
    srcs = [
        "stat_summarizer_options.h",
        "stats_calculator.cc",
    ],
    hdrs = [
        "stats_calculator.h",
    ],
    copts = tf_copts(),
    visibility = [
        "//tensorflow:internal",
    ],
)

tf_cc_test(
    name = "stats_calculator_test",
    srcs = ["stats_calculator_test.cc"],
    deps = [
        ":stats_calculator_portable",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "reporter",
    srcs = ["reporter.cc"],
    hdrs = ["reporter.h"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":test_log_proto_impl_cc",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:types",
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
    ],
    alwayslink = 1,
)

# Proto libraries.
tf_proto_library(
    name = "test_log_proto_impl",
    srcs = ["test_log.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    # Not to be used outside of tensorflow/core.
    visibility = ["//tensorflow/core:__pkg__"],
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
    ],
)
