load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "golden_baselines",
    srcs = ["golden_baselines.cc"],
    hdrs = ["golden_baselines.h"],
)

cc_library(
    name = "accuracy_budget",
    hdrs = ["accuracy_budget.h"],
)

xla_test(
    name = "intrinsic_accuracy_test",
    srcs = ["intrinsic_accuracy_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        ":accuracy_budget",
        ":golden_baselines",
        "//xla:fp_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_runner_interface",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:hlo_runner_agnostic_test_base",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)
