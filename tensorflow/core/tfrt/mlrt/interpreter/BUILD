load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//learning/brain/tfrt:__subpackages__",
        # copybara:uncomment "//smartass/brain/inference:__subpackages__",
        # copybara:uncomment "//smartass/brain/ops/tfrt_kernels:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt/translate/mlrt:__subpackages__",
        "//tensorflow/core/tfrt:__subpackages__",
    ],
)

cc_library(
    name = "context",
    srcs = ["context.cc"],
    hdrs = ["context.h"],
    deps = [
        ":attribute_span",
        ":register_span",
        ":value",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/mlrt/bytecode:executable",
        "//tensorflow/core/tfrt/mlrt/bytecode:function",
        "//tensorflow/core/tfrt/mlrt/bytecode:kernel",
        "//tensorflow/core/tfrt/mlrt/bytecode:span",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tf_runtime//:hostcontext",
    ],
)

tf_cc_test(
    name = "context_test",
    srcs = ["context_test.cc"],
    deps = [
        ":context",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "execute",
    srcs = ["execute.cc"],
    hdrs = ["execute.h"],
    deps = [
        ":context",
        ":register_span",
        ":value",
        "//tensorflow/core/tfrt/mlrt/bytecode:kernel",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "register_span",
    hdrs = ["register_span.h"],
    deps = [
        ":iterator",
        ":value",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/mlrt/bytecode:span",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "iterator",
    hdrs = ["iterator.h"],
    deps = ["//tensorflow/core/tfrt/mlrt/bytecode"],
)

cc_library(
    name = "attribute_span",
    hdrs = ["attribute_span.h"],
    deps = [
        ":iterator",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/mlrt/bytecode:span",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "value",
    hdrs = ["value.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "future",
    hdrs = ["future.h"],
    deps = [
        ":context",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@tf_runtime//:async_value",
    ],
)

cc_library(
    name = "async_handle",
    srcs = ["async_handle.cc"],
    hdrs = ["async_handle.h"],
    deps = [
        ":context",
        ":future",
        ":value",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/concurrency:async_value",
        "@tf_runtime//:async_value",
    ],
)

cc_library(
    name = "builtin_kernels",
    srcs = ["builtin_kernels.cc"],
    hdrs = ["builtin_kernels.h"],
    deps = [
        ":async_handle",
        ":context",
        ":execute",
        ":future",
        ":register_span",
        ":value",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/mlrt/bytecode:function",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "interpreter_testutil",
    testonly = 1,
    hdrs = ["interpreter_testutil.h"],
    deps = [
        ":attribute_span",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.cc"],
    deps = [
        ":async_handle",
        ":builtin_kernels",
        ":context",
        ":execute",
        ":future",
        ":interpreter_testutil",
        ":register_span",
        ":value",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/mlrt/bytecode:executable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:test_benchmark",
        "@local_xla//xla/tsl/platform:test_main",
        "@tf_runtime//:hostcontext",
    ],
)

tf_cc_test(
    name = "value_test",
    srcs = ["value_test.cc"],
    deps = [
        ":value",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "register_span_test",
    srcs = ["register_span_test.cc"],
    deps = [
        ":register_span",
        ":value",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "future_test",
    srcs = ["future_test.cc"],
    deps = [
        ":future",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/concurrency:async_value",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)
