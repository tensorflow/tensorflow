# Description:
#   Pre-set assignments data structure for BufferAssignment.

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "memory_space_assignment",
    hdrs = ["memory_space_assignment.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_value",
        "//xla/service/heap_simulator",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "memory_space_assignment_proto_cc_impl",
    visibility = ["//visibility:public"],
)
