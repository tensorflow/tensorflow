load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# Currently pybind extension shared objects must use only C API headers since
# the C API has static initializers duplicated in the Python bindings. So we
# need a second rule that omits .cc files, in
# tensorflow/python:_pywrap_parallel_device.
filegroup(
    name = "lib_headers",
    srcs = ["parallel_device_lib.h"],
)

filegroup(
    name = "lib_sources",
    srcs = ["parallel_device_lib.cc"],
)

filegroup(
    name = "device_headers",
    srcs = ["parallel_device.h"],
)

filegroup(
    name = "device_sources",
    srcs = ["parallel_device.cc"],
)

filegroup(
    name = "headers",
    srcs = [
        ":device_headers",
        ":lib_headers",
    ],
    visibility = ["//tensorflow/python:__pkg__"],
)

filegroup(
    name = "sources",
    srcs = [
        ":device_sources",
        ":lib_sources",
    ],
    visibility = ["//tensorflow/python:__pkg__"],
)

cc_library(
    name = "parallel_device",
    srcs = [":device_sources"],
    hdrs = [":device_headers"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":parallel_device_lib",
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "parallel_device_lib",
    srcs = [":lib_sources"],
    hdrs = [":lib_headers"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:tfe_cancellation_manager_internal",
        "//tensorflow/c/eager:tfe_op_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

tf_cc_test(
    name = "parallel_device_lib_test",
    srcs = ["parallel_device_lib_test.cc"],
    deps = [
        ":parallel_device_lib",
        ":parallel_device_testlib",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime/eager:context",
    ],
)

cc_library(
    name = "parallel_device_testlib",
    testonly = 1,
    srcs = ["parallel_device_testlib.cc"],
    hdrs = ["parallel_device_testlib.h"],
    deps = [
        ":parallel_device",
        ":parallel_device_lib",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "parallel_device_test",
    srcs = ["parallel_device_test.cc"],
    deps = [
        ":parallel_device",
        ":parallel_device_testlib",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "parallel_device_remote_test",
    srcs = ["parallel_device_remote_test.cc"],
    # TODO(b/136478427): Enable global heap checking when servers shut down
    # cleanly.
    args = ["--heap_check="],
    deps = [
        ":parallel_device",
        ":parallel_device_testlib",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
    ],
)
