load("//xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/service/gpu:__pkg__"],
    licenses = ["notice"],
)

cc_library(
    name = "alias_passthrough_params",
    srcs = ["alias_passthrough_params.cc"],
    hdrs = ["alias_passthrough_params.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "alias_passthrough_params_test",
    srcs = ["alias_passthrough_params_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":alias_passthrough_params",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
    ],
)
