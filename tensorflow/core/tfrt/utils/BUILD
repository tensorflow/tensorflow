load("//tensorflow:tensorflow.bzl", "if_google", "if_oss", "tf_cc_shared_test", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        "//learning/brain/google/xla/kernels/...",
        # copybara:uncomment "//learning/brain/research/pjrt/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//learning/infra/mira/distributed/...",
        # copybara:uncomment "//learning/pathways/serving/...",
        "//learning/serving/...",
        "//quality/webanswers/...",
        "//smartass/brain/inference/...",
        # copybara:uncomment "//smartass/brain/ops/...",
        "//tensorflow/c/eager/...",
        "//tensorflow/compiler/mlir/tfrt/...",
        "//tensorflow/core/runtime_fallback/...",
        "//tensorflow/core/tfrt/...",
        "//tensorflow/python/...",
        # copybara:uncomment "//tensorflow_serving/...",
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
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:profile_utils_cpu_utils",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tpu:virtual_device",
        "@com_google_absl//absl/status",
        "@local_xla//xla:status_macros",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_shared_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":utils",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:framework",
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
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/framework:tensor_shape",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//tensorflow/core/runtime_fallback/util:tensor_util",
        "//tensorflow/core/runtime_fallback/util:type_util",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
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
        "//tensorflow/c:tf_tensor",
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
        "@com_google_absl//absl/status",
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
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/concurrency:async_value",
        "@tf_runtime//:support",
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
    ],
)

cc_library(
    name = "tfrt_graph_execution_state",
    srcs = ["tfrt_graph_execution_state.cc"],
    hdrs = ["tfrt_graph_execution_state.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:upgrade_graph",
        "//tensorflow/compiler/mlir/tf2xla/api/v1:mlir_bridge_config_v1_proto_cc",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow",
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
        "//tensorflow/core/tfrt/graph_executor:config",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
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
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:while_loop",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass_registration",
        "//tensorflow/compiler/mlir/tensorflow:attribute_utils",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/grappler/utils:grappler_test",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/graph_executor:config",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "fallback_tensor",
    srcs = ["fallback_tensor.cc"],
    hdrs = ["fallback_tensor.h"],
    deps = [
        "//tensorflow/core/common_runtime:dma_helper",
        "//tensorflow/core/framework:tensor",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

tf_cc_test(
    name = "fallback_tensor_test",
    srcs = ["fallback_tensor_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":fallback_tensor",
        "//tensorflow/core/common_runtime:dma_helper",
        "//tensorflow/core/framework:tensor_shape",
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
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ],
)

cc_library(
    name = "thread_pool",
    hdrs = ["thread_pool.h"],
    visibility = [
        ":friends",
        # copybara:uncomment "//smartass/brain/server:__pkg__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:threadpool_interface",
    ],
)

cc_library(
    name = "device_variables_table",
    hdrs = ["device_variables_table.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "gpu_variables_table",
    hdrs = ["gpu_variables_table.h"],
    deps = [
        ":device_variables_table",
        ":fallback_tensor",
    ],
)

cc_library(
    name = "any_ptr",
    hdrs = ["any_ptr.h"],
)
