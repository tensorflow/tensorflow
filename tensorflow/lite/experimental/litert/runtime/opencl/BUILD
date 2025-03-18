package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "cl_command_queue",
    srcs = [
        "cl_command_queue.cc",
    ],
    hdrs = [
        "cl_command_queue.h",
    ],
    deps = [
        ":cl_context",
        ":cl_device",
        ":opencl_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@opencl_headers",
    ],
)

cc_library(
    name = "cl_device",
    srcs = [
        "cl_device.cc",
    ],
    hdrs = [
        "cl_device.h",
    ],
    deps = [
        ":opencl_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@opencl_headers",
    ],
)

cc_library(
    name = "cl_context",
    srcs = [
        "cl_context.cc",
    ],
    hdrs = [
        "cl_context.h",
    ],
    deps = [
        ":cl_device",
        ":opencl_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@opencl_headers",
    ],
)

cc_library(
    name = "cl_event",
    srcs = [
        "cl_event.cc",
    ],
    hdrs = [
        "cl_event.h",
    ],
    deps = [
        ":opencl_wrapper",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "@com_google_absl//absl/strings:str_format",
        "@opencl_headers",
    ],
)

cc_library(
    name = "opencl_wrapper",
    srcs = [
        "opencl_wrapper.cc",
    ],
    hdrs = [
        "opencl_wrapper.h",
    ],
    visibility = [
        "//tensorflow/lite/experimental/litert:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@opencl_headers",
    ],
)

cc_library(
    name = "buffer",
    srcs = [
        "buffer.cc",
    ],
    hdrs = [
        "buffer.h",
    ],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":opencl_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = ["buffer_test.cc"],
    # require GPU to run OpenCL tests.
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":buffer",
        ":cl_command_queue",
        ":cl_context",
        ":cl_device",
        ":opencl_wrapper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
