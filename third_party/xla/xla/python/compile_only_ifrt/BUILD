load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//third_party/py/jax:__subpackages__",
        "//xla/python:__subpackages__",
    ]),
)

cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:host_memory_spaces",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt",
        "//xla/python/ifrt:attribute_map",
        "//xla/python/ifrt:basic_device_list",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/pjrt_ifrt",
        "//xla/python/pjrt_ifrt:pjrt_attribute_map_util",
        "//xla/python/pjrt_ifrt:pjrt_dtype",
        "//xla/python/pjrt_ifrt:pjrt_layout",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service:computation_placer_hdr",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)
