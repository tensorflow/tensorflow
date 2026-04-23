load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "host_memory_allocator_extension",
    hdrs = ["host_memory_allocator_extension.h"],
    deps = [
        "//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "host_memory_allocator_extension_internal",
    srcs = ["host_memory_allocator_extension_internal.cc"],
    hdrs = ["host_memory_allocator_extension_internal.h"],
    deps = [
        ":host_memory_allocator_extension",
        "//xla/pjrt:host_memory_allocator",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "host_memory_allocator_interface_impl",
    srcs = ["host_memory_allocator_interface_impl.cc"],
    hdrs = ["host_memory_allocator_interface_impl.h"],
    deps = [
        ":host_memory_allocator_extension",
        "//xla/pjrt:host_memory_allocator",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_status_utils",
    ],
)
