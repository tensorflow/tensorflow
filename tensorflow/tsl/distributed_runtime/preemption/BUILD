load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/platform:build_config.bzl", "tsl_cc_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "preemption_notifier",
    srcs = ["preemption_notifier.cc"],
    hdrs = ["preemption_notifier.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:gce"],
    deps = [
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tsl_cc_test(
    name = "preemption_notifier_test",
    size = "small",
    srcs = ["preemption_notifier_test.cc"],
    deps = [
        ":preemption_notifier",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
