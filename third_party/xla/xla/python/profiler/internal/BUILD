load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/python/profiler:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "python_hooks",
    srcs = ["python_hooks.cc"],
    hdrs = ["python_hooks.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    visibility = internal_visibility([
        "//xla/backends/profiler:__subpackages__",
        "//tensorflow/python/profiler/internal:__subpackages__",
    ]),
    deps = [
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@pybind11",
    ],
    alwayslink = True,
)

cc_library(
    name = "traceme_wrapper",
    hdrs = ["traceme_wrapper.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/python:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ]),
    deps = [
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/lib:traceme_for_pybind",
        "@pybind11",
    ],
)
