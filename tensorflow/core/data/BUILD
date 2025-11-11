load(
    "//tensorflow:tensorflow.bzl",
    "if_not_mobile",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_protos_all",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

# Export files for use on Android.
exports_files([
    "captured_function.cc",
    "captured_function.h",
    "compression_utils.cc",
    "compression_utils.h",
    "dataset_utils.cc",
    "dataset_utils.h",
    "finalization_utils.cc",
    "finalization_utils.h",
    "flat_map_utils.cc",
    "flat_map_utils.h",
    "global_shuffle_utils.cc",
    "global_shuffle_utils.h",
    "metric_utils.cc",
    "metric_utils.h",
    "name_utils.cc",
    "name_utils.h",
    "rewrite_utils.cc",
    "rewrite_utils.h",
    "root_dataset.cc",
    "root_dataset.h",
    "serialization_utils.cc",
    "serialization_utils.h",
    "split_utils.cc",
    "split_utils.h",
    "stats_utils.cc",
    "stats_utils.h",
    "tf_data_memory_logger.cc",
    "tf_data_memory_logger.h",
    "tfdataz_metrics.h",
    "tfdataz_metrics.cc",
    "unbounded_thread_pool.cc",
    "unbounded_thread_pool.h",
    "utils.cc",
    "utils.h",
])

cc_library(
    name = "captured_function",
    srcs = ["captured_function.cc"],
    hdrs = ["captured_function.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset_utils",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ] + if_not_mobile([
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
    ]),
)

cc_library(
    name = "compression_utils",
    srcs = ["compression_utils.cc"],
    hdrs = ["compression_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "compression_utils_test",
    srcs = ["compression_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "requires-mem:24g",
    ],
    deps = [
        ":compression_utils",
        ":dataset_test_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "dataset_test_base",
    testonly = 1,
    srcs = ["dataset_test_base.cc"],
    hdrs = ["dataset_test_base.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset_utils",
        ":name_utils",
        ":serialization_utils",
        ":split_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/kernels:function_ops",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/framework/fixedpoint",
    ],
)

cc_library(
    name = "dataset_utils",
    srcs = ["dataset_utils.cc"],
    hdrs = ["dataset_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:regexp",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "dataset_utils_test",
    size = "small",
    srcs = ["dataset_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":compression_utils",
        ":dataset_test_base",
        ":dataset_utils",
        ":serialization_utils",
        ":test_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:str_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@local_xla//xla/tsl/util:determinism_test_util",
    ],
)

cc_library(
    name = "flat_map_utils",
    srcs = ["flat_map_utils.cc"],
    hdrs = ["flat_map_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":captured_function",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels/data:iterator_ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:refcount",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "global_shuffle_utils",
    srcs = ["global_shuffle_utils.cc"],
    hdrs = ["global_shuffle_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:framework",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "hash_utils",
    srcs = ["hash_utils.cc"],
    hdrs = ["hash_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_utils",
        ":serialization_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:regexp",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "hash_utils_test",
    size = "small",
    srcs = ["hash_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":hash_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/platform:test_main",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "tf_data_memory_logger",
    srcs = ["tf_data_memory_logger.cc"],
    hdrs = ["tf_data_memory_logger.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tfdataz_metrics",
        "//tensorflow/core/framework:model_proto_cc",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:numbers",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "metric_utils",
    srcs = ["metric_utils.cc"],
    hdrs = [
        "metric_utils.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        ":tfdataz_metrics",
        "//tensorflow/core:framework",
        "//tensorflow/core/data:utils",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:thread_annotations",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "metric_utils_test",
    size = "small",
    srcs = ["metric_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":metric_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/lib/monitoring:test_utils",
        "//tensorflow/core/platform:env",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "tfdataz_metrics",
    srcs = ["tfdataz_metrics.cc"],
    hdrs = ["tfdataz_metrics.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:thread_annotations",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "tfdataz_metrics_test",
    size = "small",
    srcs = ["tfdataz_metrics_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":tfdataz_metrics",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/util:fake_clock_env",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "name_utils",
    srcs = ["name_utils.cc"],
    hdrs = ["name_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "name_utils_test",
    size = "small",
    srcs = ["name_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":name_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "rewrite_utils",
    srcs = ["rewrite_utils.cc"],
    hdrs = ["rewrite_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_utils",
        ":hash_utils",
        ":serialization_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/grappler/optimizers/data",
        "//tensorflow/core/grappler/optimizers/data:function_utils",
        "//tensorflow/core/grappler/optimizers/data:graph_utils",
        "//tensorflow/core/platform",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "rewrite_utils_test",
    size = "small",
    srcs = ["rewrite_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":rewrite_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:function_testlib",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/ops",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "root_dataset",
    srcs = ["root_dataset.cc"],
    hdrs = ["root_dataset.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_utils",
        ":name_utils",
        ":rewrite_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:stringprintf",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "serialization_utils",
    srcs = ["serialization_utils.cc"],
    hdrs = ["serialization_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":compression_utils",
        ":dataset_utils",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "serialization_utils_test",
    size = "small",
    srcs = ["serialization_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":serialization_utils",
        ":test_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:str_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:statusor",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "split_utils",
    srcs = ["split_utils.cc"],
    hdrs = ["split_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tf_cc_test(
    name = "split_utils_test",
    size = "small",
    srcs = ["split_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_test_base",
        ":serialization_utils",
        ":split_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/framework:tensor_testutil",
    ],
)

cc_library(
    name = "snapshot_utils",
    srcs = ["snapshot_utils.cc"],
    hdrs = ["snapshot_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":name_utils",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:coding",
        "//tensorflow/core/platform:random",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "snapshot_utils_test",
    size = "small",
    srcs = ["snapshot_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":snapshot_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data/service:test_util",
    ],
)

cc_library(
    name = "standalone",
    srcs = ["standalone.cc"],
    hdrs = ["standalone.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_utils",
        ":root_dataset",
        ":serialization_utils",
        ":tf_data_memory_logger",
        ":tfdataz_metrics",
        ":unbounded_thread_pool",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/framework:graph_proto_cc",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/platform:refcount",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "standalone_save_restore_test",
    srcs = ["standalone_save_restore_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":standalone",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:test_util",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:statusor",
        "@local_xla//xla/tsl/protobuf:protos_all_cc",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "standalone_test",
    srcs = ["standalone_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":standalone",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + tf_protos_all(),
)

cc_library(
    name = "stats_utils",
    srcs = ["stats_utils.cc"],
    hdrs = ["stats_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "test_utils",
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/public:version",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "unbounded_thread_pool",
    srcs = ["unbounded_thread_pool.cc"],
    hdrs = ["unbounded_thread_pool.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "finalization_utils",
    srcs = ["finalization_utils.cc"],
    hdrs = ["finalization_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":root_dataset",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:refcount",
        "@com_google_absl//absl/status:statusor",
    ],
)

tf_cc_test(
    name = "unbounded_thread_pool_test",
    size = "small",
    srcs = ["unbounded_thread_pool_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":unbounded_thread_pool",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":utils",
        "//tensorflow/core:framework",
        "@com_google_googletest//:gtest_main",
    ],
)
