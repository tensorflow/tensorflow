load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_cloud")
load("//tensorflow/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla/python/profiler:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "python_hooks",
    srcs = ["python_hooks.cc"],
    hdrs = ["python_hooks.h"],
    compatible_with = get_compatible_with_cloud(),
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    visibility = [
        "//tensorflow/compiler/xla/backends/profiler:__subpackages__",
        "//tensorflow/core/profiler:__subpackages__",
        "//tensorflow/python/profiler/internal:__subpackages__",
    ],
    deps = [
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/utils:time_utils",
        "//tensorflow/tsl/profiler/utils:xplane_builder",
        "//tensorflow/tsl/profiler/utils:xplane_schema",
        "//tensorflow/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
    alwayslink = True,
)

cc_library(
    name = "traceme_wrapper",
    hdrs = ["traceme_wrapper.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/compiler/xla/python:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ],
    deps = [
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/lib:traceme_for_pybind",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)
