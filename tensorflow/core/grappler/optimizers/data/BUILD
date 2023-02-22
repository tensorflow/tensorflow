load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:build_config.bzl", "tf_protos_all")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core/data:__pkg__",
        "//tensorflow/core/data/service:__pkg__",
        "//tensorflow/core/grappler/optimizers/data:__subpackages__",
        "//tensorflow/core/kernels/data:__pkg__",
        "//tensorflow/core/kernels/data/experimental:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "data",
    visibility = ["//visibility:public"],
    deps = [
        ":autotune_buffer_sizes",
        ":batch_parallelization",
        ":disable_intra_op_parallelism",
        ":disable_prefetch_legacy_autotune",
        ":enable_gradient_descent",
        ":filter_fusion",
        ":filter_parallelization",
        ":inject_prefetch",
        ":make_deterministic",
        ":make_sloppy",
        ":map_and_batch_fusion",
        ":map_and_filter_fusion",
        ":map_fusion",
        ":map_parallelization",
        ":meta_optimizer",
        ":noop_elimination",
        ":parallel_batch",
        ":replicate_on_split",
        ":shuffle_and_repeat_fusion",
        ":slack",
        ":use_private_thread_pool",
    ],
)

cc_library(
    name = "auto_shard",
    srcs = ["auto_shard.cc"],
    hdrs = ["auto_shard.h"],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/kernels/data:shard_dataset_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/grappler/utils:functions",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "auto_shard_test",
    srcs = ["auto_shard_test.cc"],
    deps = [
        ":auto_shard",
        ":graph_test_utils",
        ":graph_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:function_testlib",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "autotune_buffer_sizes",
    srcs = ["autotune_buffer_sizes.cc"],
    hdrs = ["autotune_buffer_sizes.h"],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "autotune_buffer_sizes_test",
    size = "small",
    srcs = ["autotune_buffer_sizes_test.cc"],
    deps = [
        ":autotune_buffer_sizes",
        ":graph_test_utils",
        ":graph_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "batch_parallelization",
    srcs = ["batch_parallelization.cc"],
    hdrs = [
        "batch_parallelization.h",
    ],
    deps = [
        ":function_utils",
        ":graph_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "batch_parallelization_test",
    size = "small",
    srcs = ["batch_parallelization_test.cc"],
    deps = [
        ":batch_parallelization",
        ":graph_test_utils",
        ":graph_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "disable_intra_op_parallelism",
    srcs = ["disable_intra_op_parallelism.cc"],
    hdrs = ["disable_intra_op_parallelism.h"],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "disable_intra_op_parallelism_test",
    size = "small",
    srcs = ["disable_intra_op_parallelism_test.cc"],
    deps = [
        ":disable_intra_op_parallelism",
        ":graph_test_utils",
        ":graph_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "disable_prefetch_legacy_autotune",
    srcs = ["disable_prefetch_legacy_autotune.cc"],
    hdrs = ["disable_prefetch_legacy_autotune.h"],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "disable_prefetch_legacy_autotune_test",
    size = "small",
    srcs = ["disable_prefetch_legacy_autotune_test.cc"],
    deps = [
        ":disable_prefetch_legacy_autotune",
        ":graph_test_utils",
        ":graph_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "enable_gradient_descent",
    srcs = ["enable_gradient_descent.cc"],
    hdrs = ["enable_gradient_descent.h"],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "enable_gradient_descent_test",
    size = "small",
    srcs = ["enable_gradient_descent_test.cc"],
    deps = [
        ":enable_gradient_descent",
        ":graph_test_utils",
        ":graph_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "filter_fusion",
    srcs = ["filter_fusion.cc"],
    hdrs = [
        "filter_fusion.h",
    ],
    deps = [
        ":graph_utils",
        ":fusion_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/utils:topological_sort",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "filter_fusion_test",
    size = "small",
    srcs = ["filter_fusion_test.cc"],
    deps = [
        ":filter_fusion",
        ":graph_test_utils",
        ":graph_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "filter_parallelization",
    srcs = ["filter_parallelization.cc"],
    hdrs = [
        "filter_parallelization.h",
    ],
    deps = [
        ":function_utils",
        ":graph_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "filter_parallelization_test",
    size = "small",
    srcs = ["filter_parallelization_test.cc"],
    deps = [
        ":filter_parallelization",
        ":graph_test_utils",
        ":graph_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "fusion_utils",
    srcs = ["fusion_utils.cc"],
    hdrs = [
        "fusion_utils.h",
    ],
    deps = [
        ":graph_utils",
        ":function_utils",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/kernels:functional_ops",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "fusion_utils_test",
    size = "small",
    srcs = ["fusion_utils_test.cc"],
    deps = [
        ":function_utils",
        ":fusion_utils",
        ":graph_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ] + tf_protos_all(),
)

cc_library(
    name = "function_utils",
    srcs = ["function_utils.cc"],
    hdrs = [
        "function_utils.h",
    ],
    deps = [
        ":graph_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "function_utils_test",
    size = "small",
    srcs = ["function_utils_test.cc"],
    deps = [
        ":function_utils",
        ":graph_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/tools/graph_transforms:transform_utils",
    ] + tf_protos_all(),
)

cc_library(
    name = "graph_utils",
    srcs = ["graph_utils.cc"],
    hdrs = [
        "graph_utils.h",
    ],
    deps = [
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "graph_utils_test",
    size = "small",
    srcs = ["graph_utils_test.cc"],
    deps = [
        ":graph_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
)

cc_library(
    name = "graph_test_utils",
    testonly = 1,
    srcs = ["graph_test_utils.cc"],
    hdrs = [
        "graph_test_utils.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core:testlib",
    ] + tf_protos_all(),
)

cc_library(
    name = "make_deterministic",
    srcs = ["make_deterministic.cc"],
    hdrs = [
        "make_deterministic.h",
    ],
    deps = [
        ":split_utils",
        ":function_utils",
        ":graph_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "make_deterministic_test",
    size = "small",
    srcs = ["make_deterministic_test.cc"],
    deps = [
        ":function_utils",
        ":graph_test_utils",
        ":graph_utils",
        ":make_deterministic",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "split_utils",
    srcs = ["split_utils.cc"],
    hdrs = [
        "split_utils.h",
    ],
    deps = [
        ":function_utils",
        ":graph_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:framework",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:statusor",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "split_utils_test",
    size = "small",
    srcs = ["split_utils_test.cc"],
    deps = [
        ":function_utils",
        ":split_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/platform:status_matchers",
    ],
)

cc_library(
    name = "make_sloppy",
    srcs = ["make_sloppy.cc"],
    hdrs = ["make_sloppy.h"],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "make_sloppy_test",
    size = "small",
    srcs = ["make_sloppy_test.cc"],
    deps = [
        ":graph_test_utils",
        ":graph_utils",
        ":make_sloppy",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "map_and_batch_fusion",
    srcs = ["map_and_batch_fusion.cc"],
    hdrs = [
        "map_and_batch_fusion.h",
    ],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "map_and_batch_fusion_test",
    size = "small",
    srcs = ["map_and_batch_fusion_test.cc"],
    deps = [
        ":graph_utils",
        ":map_and_batch_fusion",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "map_and_filter_fusion",
    srcs = ["map_and_filter_fusion.cc"],
    hdrs = [
        "map_and_filter_fusion.h",
    ],
    deps = [
        ":graph_utils",
        ":fusion_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/utils:topological_sort",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "map_and_filter_fusion_test",
    size = "small",
    srcs = ["map_and_filter_fusion_test.cc"],
    deps = [
        ":graph_test_utils",
        ":graph_utils",
        ":map_and_filter_fusion",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "map_fusion",
    srcs = ["map_fusion.cc"],
    hdrs = [
        "map_fusion.h",
    ],
    deps = [
        ":graph_utils",
        ":fusion_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/utils:topological_sort",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "map_fusion_test",
    size = "small",
    srcs = ["map_fusion_test.cc"],
    deps = [
        ":graph_test_utils",
        ":graph_utils",
        ":map_fusion",
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/kernels:control_flow_ops",
    ],
)

cc_library(
    name = "map_parallelization",
    srcs = ["map_parallelization.cc"],
    hdrs = [
        "map_parallelization.h",
    ],
    deps = [
        ":function_utils",
        ":graph_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "map_parallelization_test",
    size = "small",
    srcs = ["map_parallelization_test.cc"],
    deps = [
        ":graph_test_utils",
        ":graph_utils",
        ":map_parallelization",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "meta_optimizer",
    srcs = ["meta_optimizer.cc"],
    hdrs = ["meta_optimizer.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/grappler/utils:functions",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ptr_util",
    ] + tf_protos_all(),
    alwayslink = 1,
)

cc_library(
    name = "noop_elimination",
    srcs = ["noop_elimination.cc"],
    hdrs = [
        "noop_elimination.h",
    ],
    deps = [
        ":function_utils",
        ":graph_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "noop_elimination_test",
    size = "small",
    srcs = ["noop_elimination_test.cc"],
    deps = [
        ":graph_utils",
        ":noop_elimination",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "optimizer_base",
    srcs = ["optimizer_base.cc"],
    hdrs = [
        "optimizer_base.h",
    ],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
    ],
)

cc_library(
    name = "parallel_batch",
    srcs = ["parallel_batch.cc"],
    hdrs = ["parallel_batch.h"],
    deps = [
        ":optimizer_base",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "parallel_batch_test",
    size = "small",
    srcs = ["parallel_batch_test.cc"],
    deps = [
        ":graph_test_utils",
        ":graph_utils",
        ":parallel_batch",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "replicate_on_split",
    srcs = ["replicate_on_split.cc"],
    hdrs = ["replicate_on_split.h"],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "replicate_on_split_test",
    size = "small",
    srcs = ["replicate_on_split_test.cc"],
    deps = [
        ":graph_test_utils",
        ":graph_utils",
        ":replicate_on_split",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "inject_prefetch",
    srcs = ["inject_prefetch.cc"],
    hdrs = ["inject_prefetch.h"],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "inject_prefetch_test",
    size = "small",
    srcs = ["inject_prefetch_test.cc"],
    deps = [
        ":graph_test_utils",
        ":graph_utils",
        ":inject_prefetch",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "shuffle_and_repeat_fusion",
    srcs = ["shuffle_and_repeat_fusion.cc"],
    hdrs = [
        "shuffle_and_repeat_fusion.h",
    ],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "shuffle_and_repeat_fusion_test",
    size = "small",
    srcs = ["shuffle_and_repeat_fusion_test.cc"],
    deps = [
        ":graph_utils",
        ":shuffle_and_repeat_fusion",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "slack",
    srcs = ["slack.cc"],
    hdrs = [
        "slack.h",
    ],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "slack_test",
    size = "small",
    srcs = ["slack_test.cc"],
    deps = [
        ":function_utils",
        ":graph_utils",
        ":slack",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/utils:functions",
        "//tensorflow/core/kernels/data",
    ],
)

cc_library(
    name = "use_private_thread_pool",
    srcs = ["use_private_thread_pool.cc"],
    hdrs = ["use_private_thread_pool.h"],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "use_private_thread_pool_test",
    size = "small",
    srcs = ["use_private_thread_pool_test.cc"],
    deps = [
        ":graph_test_utils",
        ":graph_utils",
        ":use_private_thread_pool",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)
