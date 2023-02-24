load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "async_kernel_internal",
    hdrs = ["async_kernel_internal.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "common",
    hdrs = ["common.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "task_internal",
    srcs = ["task_internal.cc"],
    hdrs = ["task_internal.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_kernel_internal",
        ":common",
        "//tensorflow/lite/core/async/interop/c:types",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_test(
    name = "task_internal_test",
    srcs = ["task_internal_test.cc"],
    deps = [
        ":async_kernel_internal",
        ":common",
        ":task_internal",
        "//tensorflow/lite/core/async/interop/c:types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "async_subgraph",
    srcs = ["async_subgraph.cc"],
    hdrs = ["async_subgraph.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_kernel_internal",
        ":common",
        ":task_internal",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/async/interop/c:types",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_test(
    name = "async_subgraph_test",
    srcs = ["async_subgraph_test.cc"],
    deps = [
        ":async_kernel_internal",
        ":async_subgraph",
        ":backend_async_kernel_interface",
        ":common",
        ":task_internal",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/core/async/interop:attribute_map_internal",
        "//tensorflow/lite/core/async/interop/c:types",
        "//tensorflow/lite/core/async/testing:mock_async_kernel",
        "//tensorflow/lite/core/async/testing:test_backend",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "backend_async_kernel_interface",
    srcs = ["backend_async_kernel_interface.cc"],
    hdrs = ["backend_async_kernel_interface.h"],
    deps = [
        ":async_kernel_internal",
        ":common",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "async_signature_runner",
    srcs = ["async_signature_runner.cc"],
    hdrs = ["async_signature_runner.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_kernel_internal",
        ":async_subgraph",
        ":common",
        ":task_internal",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/internal:signature_def",
    ],
)

cc_test(
    name = "async_signature_runner_test",
    srcs = ["async_signature_runner_test.cc"],
    deps = [
        ":async_kernel_internal",
        ":async_signature_runner",
        ":backend_async_kernel_interface",
        ":common",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:interpreter_test_util",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/core/async/c:task",
        "//tensorflow/lite/core/async/c:types",
        "//tensorflow/lite/core/async/testing:mock_async_kernel",
        "//tensorflow/lite/core/async/testing:test_backend",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "backend_async_kernel_interface_test",
    srcs = ["backend_async_kernel_interface_test.cc"],
    deps = [
        ":backend_async_kernel_interface",
        "//tensorflow/lite/core/async/testing:mock_async_kernel",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)
