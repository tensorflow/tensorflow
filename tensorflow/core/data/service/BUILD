load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
    "tf_protos_profiler_service",
)
load("//tensorflow:tensorflow.default.bzl", "cc_header_only_library", "get_compatible_with_cloud", "tf_grpc_cc_dependencies")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package_group(
    name = "data_transfer_visibility",
    packages = ["//learning/brain/google/data/service/data_transfer_protocols/..."],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "common_proto",
    srcs = ["common.proto"],
    cc_api_version = 2,
    create_java_proto = False,
    protodeps = tf_additional_all_protos(),
    visibility = [
        ":data_transfer_visibility",
        "//tensorflow:internal",
    ],
)

tf_proto_library(
    name = "dispatcher_proto",
    srcs = ["dispatcher.proto"],
    has_services = 1,
    cc_api_version = 2,
    create_java_proto = False,
    protodeps = tf_additional_all_protos() + [
        ":common_proto",
    ],
)

tf_proto_library(
    name = "worker_proto",
    srcs = ["worker.proto"],
    has_services = 1,
    cc_api_version = 2,
    create_java_proto = False,
    protodeps = tf_additional_all_protos() + [
        ":common_proto",
    ],
    visibility = [
        ":data_transfer_visibility",
        "//tensorflow:internal",
    ],
)

tf_proto_library(
    name = "export_proto",
    srcs = ["export.proto"],
    cc_api_version = 2,
    create_java_proto = False,
    protodeps = tf_additional_all_protos() + [
        ":common_proto",
    ],
)

cc_library(
    name = "auto_shard_rewriter",
    srcs = ["auto_shard_rewriter.cc"],
    hdrs = [
        "auto_shard_rewriter.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":url",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:rewrite_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers/data:auto_shard",
        "//tensorflow/core/grappler/optimizers/data:graph_utils",
        "//tensorflow/core/grappler/optimizers/data:optimizer_base",
        "//tensorflow/core/kernels/data/experimental:auto_shard_dataset_op",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "auto_shard_rewriter_test",
    size = "small",
    srcs = ["auto_shard_rewriter_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":auto_shard_rewriter",
        ":common_proto_cc",
        ":test_util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "//tensorflow/core/framework:function_testlib",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/ops",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = [
        "common.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "common_test",
    srcs = ["common_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:status_matchers",
    ],
)

cc_library(
    name = "credentials_factory",
    srcs = ["credentials_factory.cc"],
    hdrs = ["credentials_factory.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "credentials_factory_test",
    srcs = ["credentials_factory_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":credentials_factory",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "cross_trainer_cache",
    hdrs = ["cross_trainer_cache.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":logging_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:thread_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "cross_trainer_cache_test",
    size = "small",
    srcs = ["cross_trainer_cache_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":cross_trainer_cache",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:random",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "data_service_test",
    srcs = ["data_service_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":dispatcher_client",
        ":dispatcher_proto_cc",
        ":export_proto_cc",
        ":test_cluster",
        ":test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/time",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "data_transfer",
    srcs = ["data_transfer.cc"],
    hdrs = ["data_transfer.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = [
        ":data_transfer_visibility",
    ],
    deps = [
        ":common_proto_cc",
        ":worker_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:dataset_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "data_transfer_test",
    srcs = ["data_transfer_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":data_transfer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "dataset_store",
    srcs = ["dataset_store.cc"],
    hdrs = ["dataset_store.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":dispatcher_state",
        ":utils",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "dataset_store_test",
    srcs = ["dataset_store_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":dataset_store",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:graph_proto_cc",
        "@com_google_absl//absl/memory",
    ],
)

cc_grpc_library(
    name = "dispatcher_cc_grpc_proto",
    srcs = [":dispatcher_proto"],
    compatible_with = get_compatible_with_cloud(),
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    generate_mocks = True,
    grpc_only = True,
    deps = [":dispatcher_proto_cc"],
)

cc_library(
    name = "dispatcher_client",
    srcs = ["dispatcher_client.cc"],
    hdrs = ["dispatcher_client.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":credentials_factory",
        ":dispatcher_cc_grpc_proto",
        ":dispatcher_proto_cc",
        ":grpc_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/strings",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "dispatcher_client_test",
    srcs = ["dispatcher_client_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":data_transfer",
        ":dispatcher_client",
        ":test_cluster",
        ":test_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data/service/snapshot:path_utils",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "dispatcher_impl",
    srcs = ["dispatcher_impl.cc"],
    hdrs = [
        "dispatcher_impl.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":credentials_factory",
        ":dataset_store",
        ":dispatcher_proto_cc",
        ":dispatcher_state",
        ":export_proto_cc",
        ":grpc_util",
        ":journal",
        ":journal_proto_cc",
        ":split_provider",
        ":task_remover",
        ":utils",
        ":validate_utils",
        ":worker_cc_grpc_proto",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:hash_utils",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/data:standalone",
        "//tensorflow/core/data/service/snapshot:file_utils",
        "//tensorflow/core/data/service/snapshot:path_utils",
        "//tensorflow/core/data/service/snapshot:snapshot_manager",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:thread_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "dispatcher_state",
    srcs = ["dispatcher_state.cc"],
    hdrs = [
        "dispatcher_state.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":auto_shard_rewriter",
        ":common",
        ":common_proto_cc",
        ":journal",
        ":journal_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "dispatcher_state_test",
    srcs = ["dispatcher_state_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":dispatcher_state",
        ":journal_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/tsl/platform:status_matchers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grpc_dispatcher_impl",
    srcs = ["grpc_dispatcher_impl.cc"],
    hdrs = ["grpc_dispatcher_impl.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dispatcher_cc_grpc_proto",
        ":dispatcher_impl",
        ":export_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "grpc_dispatcher_impl_test",
    srcs = ["grpc_dispatcher_impl_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":credentials_factory",
        ":dispatcher_proto_cc",
        ":grpc_dispatcher_impl",
        ":server_lib",
        ":test_util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/strings",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "grpc_util",
    srcs = ["grpc_util.cc"],
    hdrs = [
        "grpc_util.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "grpc_util_test",
    srcs = ["grpc_util_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "grpc_worker_impl",
    srcs = ["grpc_worker_impl.cc"],
    hdrs = ["grpc_worker_impl.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":export_proto_cc",
        ":worker_cc_grpc_proto",
        ":worker_impl",
        ":worker_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "grpc_worker_impl_test",
    srcs = ["grpc_worker_impl_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":credentials_factory",
        ":dispatcher_proto_cc",
        ":grpc_worker_impl",
        ":server_lib",
        ":test_util",
        ":worker_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/strings",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "journal",
    srcs = ["journal.cc"],
    hdrs = ["journal.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":journal_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:regexp",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_proto_library(
    name = "journal_proto",
    srcs = ["journal.proto"],
    cc_api_version = 2,
    create_java_proto = False,
    protodeps = [
        ":common_proto",
        "//tensorflow/core:protos_all",
    ],
)

tf_cc_test(
    name = "journal_test",
    srcs = ["journal_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":journal",
        ":journal_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "logging_utils",
    srcs = ["logging_utils.cc"],
    hdrs = ["logging_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core/platform:strcat",
    ],
)

tf_cc_test(
    name = "logging_utils_test",
    srcs = ["logging_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":logging_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "py_utils",
    srcs = ["py_utils.cc"],
    hdrs = ["py_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":credentials_factory",
    ],
)

cc_library(
    name = "server_lib",
    srcs = ["server_lib.cc"],
    hdrs = ["server_lib.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    linkstatic = True,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":common_proto_cc",
        ":credentials_factory",
        ":data_transfer",
        ":export_proto_cc",
        ":grpc_dispatcher_impl",
        ":grpc_util",
        ":grpc_worker_impl",
        ":worker_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/data:utils",
        "//tensorflow/core/profiler/rpc:profiler_service_impl",
    ] + tf_grpc_cc_dependencies(),
    alwayslink = 1,
)

# This needs to be cc_header_only_library - tf_pybind_cc_library_wrapper
# does not pull in the server_lib.h header.
cc_header_only_library(
    name = "server_lib_headers_lib",
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    features = ["-parse_headers"],
    deps = [
        ":dispatcher_client",
        ":grpc_util",
        ":server_lib",
    ],
)

cc_library(
    name = "split_provider",
    srcs = ["split_provider.cc"],
    hdrs = [
        "split_provider.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":dispatcher_client",
        ":grpc_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/data:standalone",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:thread_annotations",
    ],
)

cc_library(
    name = "task_remover",
    srcs = ["task_remover.cc"],
    hdrs = ["task_remover.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "task_runner",
    srcs = ["task_runner.cc"],
    hdrs = ["task_runner.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":cross_trainer_cache",
        ":data_transfer",
        ":logging_utils",
        ":thread_safe_buffer",
        ":worker_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:standalone",
    ],
)

tf_cc_test(
    name = "task_runner_test",
    srcs = ["task_runner_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":data_transfer",
        ":task_runner",
        ":worker_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:tstring",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "test_cluster",
    testonly = True,
    srcs = ["test_cluster.cc"],
    hdrs = ["test_cluster.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":data_transfer",
        ":dispatcher_client",
        ":dispatcher_proto_cc",
        ":export_proto_cc",
        ":server_lib",
        ":test_util",
        ":worker_client",
        ":worker_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = [
        "test_util.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    data = ["//tensorflow/core/data/service/testdata"],
    deps = [
        ":common_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:function_testlib",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "test_util_test",
    srcs = ["test_util_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":test_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:standalone",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "thread_safe_buffer",
    hdrs = ["thread_safe_buffer.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
    ],
)

tf_cc_test(
    name = "thread_safe_buffer_test",
    size = "small",
    srcs = ["thread_safe_buffer_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    shard_count = 3,
    deps = [
        ":thread_safe_buffer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "url",
    srcs = ["url.cc"],
    hdrs = ["url.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core/platform:regexp",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "url_test",
    size = "small",
    srcs = ["url_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":url",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "validate_utils",
    srcs = ["validate_utils.cc"],
    hdrs = ["validate_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
    ],
)

tf_cc_test(
    name = "validate_utils_test",
    size = "small",
    srcs = ["validate_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":validate_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_shape_proto_cc",
        "//tensorflow/core/platform:status_matchers",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
    ],
)

cc_grpc_library(
    name = "worker_cc_grpc_proto",
    srcs = [":worker_proto"],
    compatible_with = get_compatible_with_cloud(),
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    generate_mocks = True,
    grpc_only = True,
    deps = [":worker_proto_cc"],
)

cc_library(
    name = "worker_client",
    srcs = ["worker_client.cc"],
    hdrs = ["worker_client.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":credentials_factory",
        ":data_transfer",
        ":grpc_util",
        ":worker_cc_grpc_proto",
        ":worker_impl",
        ":worker_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/framework:dataset_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "worker_client_test",
    size = "small",
    srcs = ["worker_client_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":data_transfer",
        ":dispatcher_client",
        ":dispatcher_proto_cc",
        ":server_lib",
        ":test_cluster",
        ":test_util",
        ":worker_client",
        ":worker_impl",
        ":worker_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "worker_impl",
    srcs = ["worker_impl.cc"],
    hdrs = [
        "worker_impl.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":auto_shard_rewriter",
        ":common",
        ":common_proto_cc",
        ":credentials_factory",
        ":data_transfer",
        ":dispatcher_cc_grpc_proto",
        ":dispatcher_client",
        ":dispatcher_proto_cc",
        ":export_proto_cc",
        ":grpc_util",
        ":split_provider",
        ":task_runner",
        ":utils",
        ":worker_proto_cc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:standalone",
        "//tensorflow/core/data/service/snapshot:path_utils",
        "//tensorflow/core/data/service/snapshot:snapshot_split_provider",
        "//tensorflow/core/data/service/snapshot:snapshot_stream_writer",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/platform:status_to_from_proto",
        "//tensorflow/tsl/protobuf:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "worker_impl_test",
    srcs = ["worker_impl_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":test_cluster",
        ":worker_impl",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + tf_protos_profiler_service(),
)
