load("//xla/tsl:tsl.bzl", "if_windows", "internal_visibility")
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
        ":traceme_state",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:macros",
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@pybind11",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "traceme_state",
    srcs = ["traceme_state.cc"],
    hdrs = ["traceme_state.h"],
    copts = tf_profiler_copts() + if_windows(["/DTF_COMPILE_LIBRARY"]),
    visibility = internal_visibility([
        "//tensorflow/python/profiler/internal:__pkg__",
    ]),
    deps = [
        "//xla/tsl/platform:macros",
    ],
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
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/strings",
        "@pybind11",
        "@tsl//tsl/profiler/lib:traceme_for_pybind",
    ],
)
