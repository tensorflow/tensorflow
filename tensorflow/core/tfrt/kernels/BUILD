load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_portable",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/core/tfrt/...",
    ],
)

cc_library(
    name = "ifrt_program_ops",
    srcs = ["ifrt_program_ops.cc"],
    hdrs = ["ifrt_program_ops.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/tfrt/ifrt:ifrt_executable_registry",
        "//tensorflow/core/tfrt/ifrt:ifrt_serving_executable",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "ifrt_program_ops_test",
    srcs = ["ifrt_program_ops_test.cc"],
    data = [
        "//tensorflow/core/tfrt/ifrt/testdata",
    ],
    tags = ["no_oss"],
    deps = [
        ":ifrt_program_ops",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:tensor_matcher",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/lib/gtl:cleanup",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tfrt/ifrt:ifrt_executable_registry",
        "//tensorflow/core/tfrt/ifrt:ifrt_serving_executable_test_util",
        "//tensorflow/core/tfrt/ops:ifrt_program_ops_op_lib",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/python/ifrt",
        "@local_xla//xla/python/ifrt:test_util",
        "@local_xla//xla/python/pjrt_ifrt",
        "@local_xla//xla/python/pjrt_ifrt:tfrt_cpu_client_test_lib",
        "@local_xla//xla/tsl/framework:serving_device_selector",
        "@local_xla//xla/tsl/framework/test_util:mock_serving_device_selector",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "stream_ops",
    srcs = ["stream_ops.cc"],
    hdrs = ["stream_ops.h"],
    deps = [
        ":stream_ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/tfrt/runtime:stream",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stream_ops_util_constants",
    hdrs = ["stream_ops_util_constants.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "stream_ops_util",
    srcs = ["stream_ops_util.cc"],
    hdrs = ["stream_ops_util.h"],
    deps = [
        ":stream_ops_util_constants",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "stream_ops_util_test",
    srcs = ["stream_ops_util_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":stream_ops_util",
        ":stream_ops_util_constants",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:tensor_matcher",
        "//tensorflow/core/framework:tensor_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)
