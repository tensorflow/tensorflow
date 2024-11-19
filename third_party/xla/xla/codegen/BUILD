load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "kernel_emitter",
    hdrs = ["kernel_emitter.h"],
    deps = [
        ":kernel_spec",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    deps = [
        "//xla/runtime:buffer_use",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)
