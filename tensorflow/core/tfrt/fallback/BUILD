load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_mobile",
    "if_not_mobile",
    "tf_cc_test",
    "tf_features_nolayering_check_if_ios",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load("//tensorflow:tensorflow.default.bzl", "tf_cuda_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
        "//tensorflow/compiler/mlir/tfrt/...",
        "//tensorflow/core/...",
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        # copybara:uncomment "//learning/brain/mobile/lite/delegates/tfmrt/...",
    ],
)

cc_library(
    name = "fallback_state",
    srcs = ["fallback_state.cc"],
    hdrs = ["fallback_state.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:gce"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/framework:graph_proto_cc",
    ],
)

tf_cc_test(
    name = "fallback_state_test",
    srcs = ["fallback_state_test.cc"],
    deps = [
        ":fallback_state",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "op_kernel_runner",
    srcs = ["op_kernel_runner.cc"],
    hdrs = ["op_kernel_runner.h"],
    features = tf_features_nolayering_check_if_ios(),
    visibility = [
        # copybara:uncomment "//tensorflow/core/runtime_fallback:internal",
        # copybara:uncomment "//tensorflow/core/tfrt/eager:__pkg__",
        "//tensorflow/lite/delegates/flex:__pkg__",
    ],
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core:framework",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:op_def_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
    ]),
)

cc_library(
    name = "op_kernel_runner_cache",
    srcs = ["op_kernel_runner_cache.cc"],
    hdrs = ["op_kernel_runner_cache.h"],
    deps = [
        ":op_kernel_runner",
        "//tensorflow/core/framework:node_def_util",
        "@com_google_absl//absl/strings",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "cost_recorder",
    srcs = ["cost_recorder.cc"],
    hdrs = ["cost_recorder.h"],
    deps = [
        ":op_cost_map_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/util:env_var",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tf_runtime//:hostcontext",
    ],
)

tf_cc_test(
    name = "cost_recorder_test",
    srcs = ["cost_recorder_test.cc"],
    deps = [
        ":cost_recorder",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cuda_cc_test(
    name = "op_kernel_runner_test",
    size = "small",
    srcs = ["op_kernel_runner_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":op_kernel_runner",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_static(
        [
            "//tensorflow/core/common_runtime:function",
        ],
        [],
    ),
)

tf_proto_library(
    name = "op_cost_map_proto",
    srcs = ["op_cost_map.proto"],
    cc_api_version = 2,
)
