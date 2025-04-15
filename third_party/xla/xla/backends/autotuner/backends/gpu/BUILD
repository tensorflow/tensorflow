load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_backend",
    hdrs = ["gpu_backend.h"],
    deps = [
        "//xla/backends/autotuner:backend",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)
