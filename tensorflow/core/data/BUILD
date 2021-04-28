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

# Export a few files for use on Android.
exports_files([
    "captured_function.cc",
    "captured_function.h",
    "dataset_utils.cc",
    "dataset_utils.h",
    "name_utils.cc",
    "name_utils.h",
    "split_utils.h",
    "split_utils.cc",
    "stats_utils.cc",
    "stats_utils.h",
])

cc_library(
    name = "captured_function",
    srcs = ["captured_function.cc"],
    hdrs = ["captured_function.h"],
    deps = [
        ":dataset_utils",
        ":single_threaded_executor",
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
    deps = [
        ":name_utils",
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
        "//tensorflow/core/data:dataset_utils",
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
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/grappler/optimizers/data",
        "//tensorflow/core/grappler/optimizers/data:function_utils",
        "//tensorflow/core/grappler/optimizers/data:graph_utils",
    ],
)

cc_library(
    name = "serialization_utils",
    srcs = ["serialization_utils.cc"],
    hdrs = ["serialization_utils.h"],
    deps = [
        ":captured_function",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/core:status",
    ],
)

cc_library(
    name = "single_threaded_executor",
    srcs = ["single_threaded_executor.cc"],
    hdrs = ["single_threaded_executor.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime:entry",
        "//tensorflow/core/common_runtime:local_executor_params",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "single_threaded_executor_test",
    size = "small",
    srcs = ["single_threaded_executor_test.cc"],
    deps = [
        ":single_threaded_executor",
        "//tensorflow/core:bitwise_ops_op_lib",
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:random_ops_op_lib",
        "//tensorflow/core:spectral_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:state",
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
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:session_options",
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
