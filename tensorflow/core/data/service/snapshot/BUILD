# Distributed snapshot library.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "tf_grpc_cc_dependencies")
load("//tensorflow/core/platform:build_config.bzl", "tf_protos_profiler_service")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

tf_cc_test(
    name = "distributed_snapshot_test",
    srcs = ["distributed_snapshot_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":path_utils",
        ":snapshot_reader",
        ":test_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/data/service:dispatcher_client",
        "//tensorflow/core/data/service:test_cluster",
        "//tensorflow/core/data/service:test_util",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:tstring",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "file_utils",
    srcs = ["file_utils.cc"],
    hdrs = ["file_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:random",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "file_utils_test",
    srcs = ["file_utils_test.cc"],
    deps = [
        ":file_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/data/service:test_util",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:protos_all_cc",
    ],
)

cc_library(
    name = "path_utils",
    srcs = ["path_utils.cc"],
    hdrs = ["path_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "path_utils_test",
    srcs = ["path_utils_test.cc"],
    deps = [
        ":path_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/protobuf:protos_all_cc",
    ],
)

cc_library(
    name = "snapshot_split_provider",
    srcs = ["snapshot_split_provider.cc"],
    hdrs = ["snapshot_split_provider.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":file_utils",
        ":path_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/data/service:dispatcher_client",
        "//tensorflow/core/data/service:dispatcher_proto_cc",
        "//tensorflow/core/data/service:grpc_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:thread_annotations",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "snapshot_split_provider_test",
    srcs = ["snapshot_split_provider_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":file_utils",
        ":path_utils",
        ":snapshot_split_provider",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:dispatcher_client",
        "//tensorflow/core/data/service:test_util",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/protobuf:protos_all_cc",
    ],
)

cc_library(
    name = "snapshot_manager",
    srcs = ["snapshot_manager.cc"],
    hdrs = ["snapshot_manager.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":file_utils",
        ":path_utils",
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:dispatcher_proto_cc",
        "//tensorflow/core/data/service:split_provider",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_to_from_proto",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "snapshot_manager_test",
    size = "small",
    srcs = ["snapshot_manager_test.cc"],
    deps = [
        ":path_utils",
        ":snapshot_manager",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:dispatcher_proto_cc",
        "//tensorflow/core/data/service:test_util",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:status_to_from_proto",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:protos_all_cc",
    ],
)

cc_library(
    name = "snapshot_reader",
    srcs = ["snapshot_reader.cc"],
    hdrs = ["snapshot_reader.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":file_utils",
        ":path_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:refcount",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "snapshot_stream_writer",
    srcs = ["snapshot_stream_writer.cc"],
    hdrs = ["snapshot_stream_writer.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":file_utils",
        ":path_utils",
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:task_runner",
        "//tensorflow/core/data/service:worker_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:regexp",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:thread_annotations",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "snapshot_stream_writer_checkpoint_test",
    size = "small",
    srcs = ["snapshot_stream_writer_checkpoint_test.cc"],
    deps = [
        ":path_utils",
        ":snapshot_reader",
        ":snapshot_stream_writer",
        ":test_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:task_runner",
        "//tensorflow/core/data/service:test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "snapshot_stream_writer_test",
    size = "small",
    srcs = ["snapshot_stream_writer_test.cc"],
    deps = [
        ":path_utils",
        ":snapshot_stream_writer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/data:standalone",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:task_runner",
        "//tensorflow/core/data/service:test_util",
        "//tensorflow/tsl/lib/monitoring:cell_reader",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "test_utils",
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":file_utils",
        ":path_utils",
        ":snapshot_stream_writer",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/data:standalone",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:task_runner",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:dataset_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:status",
    ],
)
