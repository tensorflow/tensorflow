load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "backend",
    hdrs = ["backend.h"],
    visibility = [
        "//xla/backends/autotuner:__subpackages__",
    ],
    deps = [
        "//third_party/protobuf",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)
