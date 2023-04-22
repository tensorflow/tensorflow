load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_not_mobile",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
    "tf_protos_all",
)

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

# Export files for use on Android.
exports_files([
    "captured_function.cc",
    "captured_function.h",
    "dataset_utils.cc",
    "dataset_utils.h",
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
    "unbounded_thread_pool.cc",
    "unbounded_thread_pool.h",
])

cc_library(
    name = "captured_function",
    srcs = ["captured_function.cc"],
    hdrs = ["captured_function.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset_utils",
        ":stats_utils",
        "@com_google_absl//absl/time",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:variable_ops",
        "//tensorflow/core/profiler/lib:traceme",
    ] + if_not_mobile([
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
    ]),
)

cc_library(
    name = "compression_utils",
    srcs = ["compression_utils.cc"],
    hdrs = ["compression_utils.h"],
    deps = [
        ":dataset_proto_cc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "compression_utils_test",
    srcs = ["compression_utils_test.cc"],
    deps = [
        ":compression_utils",
        ":dataset_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_proto_library(
    name = "dataset_proto",
    srcs = ["dataset.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
)

cc_library(
    name = "dataset_test_base",
    testonly = 1,
    srcs = ["dataset_test_base.cc"],
    hdrs = ["dataset_test_base.h"],
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
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dataset_utils",
    srcs = ["dataset_utils.cc"],
    hdrs = ["dataset_utils.h"],
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
    ],
)

tf_cc_test(
    name = "dataset_utils_test",
    size = "small",
    srcs = ["dataset_utils_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":serialization_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:str_util",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "hash_utils",
    srcs = ["hash_utils.cc"],
    hdrs = ["hash_utils.h"],
    deps = [
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:regexp",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "hash_utils_test",
    size = "small",
    srcs = ["hash_utils_test.cc"],
    deps = [
        ":hash_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "name_utils",
    srcs = ["name_utils.cc"],
    hdrs = ["name_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "name_utils_test",
    size = "small",
    srcs = ["name_utils_test.cc"],
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
    deps = [
        ":dataset_utils",
        ":hash_utils",
        ":serialization_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_view",
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
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "rewrite_utils_test",
    size = "small",
    srcs = ["rewrite_utils_test.cc"],
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
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "root_dataset",
    srcs = ["root_dataset.cc"],
    hdrs = ["root_dataset.h"],
    deps = [
        ":dataset_utils",
        ":name_utils",
        ":rewrite_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:stringprintf",
    ],
)

cc_library(
    name = "serialization_utils",
    srcs = ["serialization_utils.cc"],
    hdrs = ["serialization_utils.h"],
    deps = [
        ":dataset_utils",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core/lib/core:status",
    ],
)

tf_cc_test(
    name = "serialization_utils_test",
    size = "small",
    srcs = ["serialization_utils_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":serialization_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:str_util",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "split_utils",
    srcs = ["split_utils.cc"],
    hdrs = ["split_utils.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "split_utils_test",
    size = "small",
    srcs = ["split_utils_test.cc"],
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
    deps = [
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/platform:coding",
        "//tensorflow/core/platform:random",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "snapshot_utils_test",
    size = "small",
    srcs = ["snapshot_utils_test.cc"],
    deps = [
        ":snapshot_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "standalone",
    srcs = ["standalone.cc"],
    hdrs = ["standalone.h"],
    deps = [
        ":root_dataset",
        ":unbounded_thread_pool",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:session_options",
        "//tensorflow/core/platform:refcount",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "standalone_test",
    srcs = ["standalone_test.cc"],
    deps = [
        ":standalone",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + tf_protos_all(),
)

cc_library(
    name = "stats_utils",
    srcs = ["stats_utils.cc"],
    hdrs = ["stats_utils.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "unbounded_thread_pool",
    srcs = ["unbounded_thread_pool.cc"],
    hdrs = ["unbounded_thread_pool.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "unbounded_thread_pool_test",
    size = "small",
    srcs = ["unbounded_thread_pool_test.cc"],
    deps = [
        ":unbounded_thread_pool",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
