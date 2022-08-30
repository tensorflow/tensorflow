load("//tensorflow:tensorflow.bzl", "if_google", "if_oss", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        # copybara:uncomment "//learning/brain/research/pjrt/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//smartass/brain/ops/...",
        "//tensorflow/c/eager/...",
        "//tensorflow/compiler/mlir/tfrt/...",
        "//tensorflow/core/runtime_fallback/...",
        "//tensorflow/core/tfrt/...",
        "//tensorflow/python/...",
        # copybara:uncomment "//tensorflow_serving/batching/google/...",
        # copybara:uncomment "//tensorflow_serving/servables/tensorflow/google/...",
        # copybara:uncomment "//third_party/tf_runtime_google/...",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "dtype.def",
        "utils.h",
    ],
    deps = [
        ":error_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/tfrt/eager:virtual_device",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tpu:virtual_device",
        "@com_google_absl//absl/status",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core_no_xla",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//cpp_tests:common",
    ],
)

cc_library(
    name = "tensor_util",
    srcs = ["tensor_util.cc"],
    hdrs = [
        "dtype.def",
        "tensor_util.h",
    ],
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/framework:tensor_shape",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//tensorflow/core/runtime_fallback/util:tensor_util",
        "//tensorflow/core/runtime_fallback/util:type_util",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": ["//tensorflow/core/framework:tensor"],
    }),
)

tf_cc_test(
    name = "tensor_util_test",
    srcs = ["tensor_util_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":tensor_util",
        "//tensorflow/core:framework",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//cpp_tests:common",
    ],
)

cc_library(
    name = "error_util",
    srcs = [
        "error_util.cc",
    ],
    hdrs = [
        "error_type.def",
        "error_util.h",
    ],
    deps = [
        "//tensorflow/core/platform:status",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "error_util_test",
    srcs = ["error_util_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":error_util",
        "//tensorflow/core/platform:status",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
        "@tf_runtime//cpp_tests:common",
    ],
)

cc_library(
    name = "graph_partition",
    srcs = ["graph_partition.cc"],
    hdrs = ["graph_partition.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/common_runtime:graph_constructor",
        "//tensorflow/core/common_runtime:partitioning_utils",
        "//tensorflow/core/grappler:utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "graph_partition_test",
    srcs = ["graph_partition_test.cc"],
    deps = [
        ":graph_partition",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/common_runtime:inline_function_utils",
        "//tensorflow/core/common_runtime:placer",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/grappler/utils:grappler_test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tfrt_graph_execution_state",
    srcs = ["tfrt_graph_execution_state.cc"],
    hdrs = ["tfrt_graph_execution_state.h"],
    deps = [
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/jit:compilation_passes",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:upgrade_graph",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:op_def_proto_cc",
        "//tensorflow/core/framework:versions_proto_cc",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "tfrt_graph_execution_state_test",
    srcs = ["tfrt_graph_execution_state_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # b/169705709, no protobuf matchers in OSS.
    deps = [
        ":tfrt_graph_execution_state",
        "//tensorflow/cc:array_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:math_ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/cc:while_loop",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/tf2xla/cc:xla_jit_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/grappler/utils:grappler_test",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fallback_tensor",
    srcs = ["fallback_tensor.cc"],
    hdrs = ["fallback_tensor.h"],
    deps = [
        "//tensorflow/core/common_runtime:dma_helper",
        "//tensorflow/core/framework:tensor",
        "@com_google_absl//absl/types:variant",
    ],
)

tf_cc_test(
    name = "fallback_tensor_test",
    srcs = ["fallback_tensor_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":fallback_tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bridge_graph_analysis",
    hdrs = ["bridge_graph_analysis.h"],
    visibility = [
        "//tensorflow/core/tfrt/graph_executor:__pkg__",
    ],
    deps = if_google([
        "//learning/brain/mlir/bridge:graph_analysis",
        "//tensorflow/core/platform:enable_tf2_utils",
    ]) + [
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core:core_cpu_base",
    ],
)

cc_library(
    name = "thread_pool",
    hdrs = ["thread_pool.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:threadpool_interface",
    ],
)
