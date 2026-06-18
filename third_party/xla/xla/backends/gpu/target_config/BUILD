load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/util:cc_embed_data.bzl", "cc_embed_data")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_gpu_specs",
    data = glob(["specs/*.txtpb"]),
)

exports_files(glob([
    "specs/*.txtpb",
]))

cc_embed_data(
    name = "embed_gpu_specs",
    srcs = glob(["specs/*.txtpb"]),
    outs = [
        "embed_gpu_specs.cc",
        "embed_gpu_specs.h",
    ],
    compatible_with = get_compatible_with_portable() + if_google(["//buildenv/target:libtpu"]),
    embedopts = ["--namespace=xla::gpu"],
    flatten = True,
)

cc_library(
    name = "target_config",
    srcs = ["target_config.cc"],
    hdrs = ["target_config.h"],
    compatible_with = get_compatible_with_portable() + if_google(["//buildenv/target:libtpu"]),
    deps = [
        ":embed_gpu_specs",
        "//xla:status_macros",
        "//xla:xla_proto_cc",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
    ],
)

xla_cc_test(
    name = "target_config_test",
    srcs = ["target_config_test.cc"],
    deps = [
        ":target_config",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "gpu_topology_utils",
    srcs = ["gpu_topology_utils.cc"],
    hdrs = ["gpu_topology_utils.h"],
    compatible_with = get_compatible_with_portable() + if_google(["//buildenv/target:libtpu"]),
    visibility = internal_visibility([
        "//xla/pjrt/gpu:legacy_gpu_topology_users",
        ":__subpackages__",
    ]),
    deps = [
        "//xla/service:gpu_topology",
        "//xla/service:gpu_topology_proto_cc",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "gpu_topology_utils_test",
    srcs = ["gpu_topology_utils_test.cc"],
    deps = [
        ":gpu_topology_utils",
        ":target_config",
        "//xla/service:gpu_topology",
        "//xla/service:gpu_topology_proto_cc",
        "//xla/stream_executor/cuda:cuda_compute_capability_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "embedded_target_config_test",
    srcs = ["embedded_target_config_test.cc"],
    backend_tags = {
        "h100": ["full"],
        "b200": ["full"],
        "gb200": ["full"],
        "gb300": ["full"],
    },
    backends = [
        "p100",
        "v100",
        "a100",
        "h100",
        "b200",
        "gb200",
        "gb300",
        "amdgpu_any",
        "intelgpu_any",
    ],
    data = [":all_gpu_specs"],
    tags = [
        # TODO(intel-tf): Implement device description for oneAPI and enable this test.
        "no-oneapi",
    ],
    deps = [
        "//xla/service:platform_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_test(
    name = "embedded_gpu_topology_test",
    srcs = ["embedded_gpu_topology_test.cc"],
    backend_tags = {
        "b200": ["full"],
        "gb200": ["full"],
        "gb300": ["full"],
    },
    backends = [
        "b200",
        "gb200",
        "gb300",
    ],
    tags = [
        # This is testing whether the system configs are up to date and the test would fail on any
        # machine with a slightly different spec.
        "no_oss",
    ],
    deps = [
        ":target_config",
        "//xla:debug_options_flags",
        "//xla/backends/cpu:target_machine_options",
        "//xla/service:gpu_topology",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
    ],
)
