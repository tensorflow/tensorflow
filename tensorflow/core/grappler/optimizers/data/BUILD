licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_protos_all")

package(default_visibility = [
    "//tensorflow/core/grappler/optimizers/data:__subpackages__",
    "//tensorflow/core/kernels/data:__pkg__",
])

cc_library(
    name = "data",
    visibility = ["//visibility:public"],
    deps = [
        ":filter_fusion",
        ":hoist_random_uniform",
        ":latency_all_edges",
        ":make_numa_aware",
        ":make_sloppy",
        ":map_and_batch_fusion",
        ":map_and_filter_fusion",
        ":map_fusion",
        ":map_parallelization",
        ":map_vectorization",
        ":meta_optimizer",
        ":noop_elimination",
        ":shuffle_and_repeat_fusion",
        ":filter_with_random_uniform_fusion",
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
    ],
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
    name = "hoist_random_uniform",
    srcs = ["hoist_random_uniform.cc"],
    hdrs = [
        "hoist_random_uniform.h",
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
        "//tensorflow/core:lib_internal",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "hoist_random_uniform_test",
    srcs = ["hoist_random_uniform_test.cc"],
    deps = [
        ":graph_test_utils",
        ":graph_utils",
        ":hoist_random_uniform",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ] + tf_protos_all(),
)

cc_library(
    name = "latency_all_edges",
    srcs = ["latency_all_edges.cc"],
    hdrs = [
        "latency_all_edges.h",
    ],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "latency_all_edges_test",
    srcs = ["latency_all_edges_test.cc"],
    deps = [
        ":graph_utils",
        ":latency_all_edges",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "make_numa_aware",
    srcs = ["make_numa_aware.cc"],
    hdrs = ["make_numa_aware.h"],
    deps = [
        ":graph_utils",
        ":optimizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
    alwayslink = 1,
)

tf_cc_test(
    name = "make_numa_aware_test",
    srcs = ["make_numa_aware_test.cc"],
    deps = [
        ":graph_test_utils",
        ":graph_utils",
        ":make_numa_aware",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "make_sloppy",
    srcs = ["make_sloppy.cc"],
    hdrs = ["make_sloppy.h"],
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
    name = "make_sloppy_test",
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
    name = "map_vectorization",
    srcs = ["map_vectorization.cc"],
    hdrs = [
        "map_vectorization.h",
    ],
    deps = [
        ":function_utils",
        ":graph_utils",
        ":optimizer_base",
        ":vectorization_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:mutable_graph_view",
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
    name = "map_vectorization_test",
    srcs = ["map_vectorization_test.cc"],
    deps = [
        ":graph_utils",
        ":map_vectorization",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:bitwise_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:spectral_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/utils:topological_sort",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels/data",
    ],
)

cc_library(
    name = "meta_optimizer",
    srcs = ["meta_optimizer.cc"],
    hdrs = ["meta_optimizer.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:arithmetic_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/grappler/optimizers:dependency_optimizer",
        "//tensorflow/core/grappler/optimizers:function_optimizer",
        "//tensorflow/core/grappler/optimizers:model_pruner",
        "//tensorflow/core/grappler/optimizers:shape_optimizer",
        "//tensorflow/core/grappler:grappler_item",
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
        "//tensorflow/core:metrics",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
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
    name = "vectorization_utils",
    srcs = ["vectorization_utils.cc"],
    hdrs = [
        "vectorization_utils.h",
    ],
    deps = [
        ":function_utils",
        ":graph_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/cc:ops",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/optimizers/data/vectorization",
        "//tensorflow/core/grappler/utils:functions",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "vectorization_utils_test",
    srcs = ["vectorization_utils_test.cc"],
    deps = [
        ":graph_utils",
        ":function_utils",
        ":vectorization_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        # For ops we need registered
        "//tensorflow/core/kernels/data:dataset_ops",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:logging_ops",
        "//tensorflow/core:logging_ops_op_lib",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core:spectral_ops_op_lib",
        "//tensorflow/core:bitwise_ops_op_lib",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core/kernels:nn",
        "//tensorflow/core:sendrecv_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:mkl_nn_ops_op_lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core/kernels:parsing",
        "//tensorflow/core:parsing_ops_op_lib",
        "//tensorflow/tools/graph_transforms:transform_utils",
    ] + tf_protos_all(),
)

cc_library(
    name = "filter_with_random_uniform_fusion",
    srcs = ["filter_with_random_uniform_fusion.cc"],
    hdrs = [
        "filter_with_random_uniform_fusion.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "filter_with_random_uniform_fusion_test",
    srcs = ["filter_with_random_uniform_fusion_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_utils",
        ":filter_with_random_uniform_fusion",
        ":graph_test_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
    ],
)
