load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable")
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
    name = "host_offloading_allocator",
    hdrs = ["host_offloading_allocator.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "host_offloading_buffer",
    hdrs = ["host_offloading_buffer.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)
