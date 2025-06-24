load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "mock_device_time_measurement",
    testonly = True,
    srcs = [
        "mock_device_time_measurement.cc",
        "//xla/pjrt/profiling:device_time_measurement.h",
    ],
    hdrs = ["mock_device_time_measurement.h"],
    # copybara:uncomment_begin(google-only)
    # compatible_with = ["//buildenv/target:non_prod"],
    # copybara:uncomment_end
    deps = [
        "//xla/pjrt:pjrt_compiler",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
