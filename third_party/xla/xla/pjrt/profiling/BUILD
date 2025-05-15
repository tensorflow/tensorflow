# Device runtime profiling.
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/google/runtime:__subpackages__",
        "//learning/brain/tpu/runtime/hal/internal:__subpackages__",
        "//learning/serving/model_servers:__subpackages__",
        "//xla:internal",
    ]),
    licenses = ["notice"],
)

exports_files(
    [
        "device_time_measurement.h",
        "no_op_device_time_measurement.h",
        "profiling_context.h",
    ],
    visibility = internal_visibility([
        "//learning/brain/google/runtime:__subpackages__",
        "//learning/brain/tpu/runtime/hal/internal:__subpackages__",
        "//tensorflow:__subpackages__",
    ]),
)

cc_library(
    name = "no_op_device_time_measurement",
    srcs = [
        "device_time_measurement.h",
        "no_op_device_time_measurement.cc",
        "no_op_device_time_measurement.h",
    ],
    # copybara:uncomment_begin(google-only)
    # compatible_with = ["//buildenv/target:non_prod"],
    # copybara:uncomment_end
    deps = [
        "//xla/pjrt:pjrt_compiler",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "profiling_context_no_op",
    srcs = [
        "profiling_context.h",
        "profiling_context_no_op.cc",
        "profiling_context_no_op.h",
    ],
    # copybara:uncomment_begin(google-only)
    # compatible_with = ["//buildenv/target:non_prod"],
    # copybara:uncomment_end
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "device_time_measurement",
    # copybara:uncomment_begin(google-only)
    # compatible_with = ["//buildenv/target:non_prod"],
    # copybara:uncomment_end
    textual_hdrs = ["device_time_measurement.h"],
    deps = [
        # copybara:comment_begin(oss-only)
        ":no_op_device_time_measurement",
        # copybara:comment_end
        # copybara:uncomment_begin(google-only)
        # "//learning/brain/google/runtime:device_runtime_profiling",  # buildcleaner: keep
        # copybara:uncomment_end
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//xla/pjrt:pjrt_compiler",
    ],
)

cc_library(
    name = "profiling_context",
    # copybara:uncomment_begin(google-only)
    # compatible_with = ["//buildenv/target:non_prod"],
    # copybara:uncomment_end
    textual_hdrs = ["profiling_context.h"],
    deps = [
        # copybara:comment_begin(oss-only)
        ":profiling_context_no_op",
        # copybara:comment_end
        # copybara:uncomment_begin(google-only)
        # "//learning/brain/google/runtime:profiling_context",
        # copybara:uncomment_end
        "@llvm-project//llvm:Support",
    ],
)
