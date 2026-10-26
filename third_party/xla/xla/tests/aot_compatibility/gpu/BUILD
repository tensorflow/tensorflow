load("@rules_cc//cc:defs.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

_GPU_TARGETS = [
    "h100",
    "b200",
    "gb200",
    "gb300",
]

_DISABLED_GPU_TARGETS = [
    # TODO(b/491194726): Serialized executable compute capability 6.0 does not match sm_10.3a.
    "gb300",
]

filegroup(
    name = "test_dummy_test_data",
    testonly = True,
    srcs = glob(["executables/test_dummy_test/**"]),
    visibility = ["//xla/tests/aot_compatibility:__pkg__"],
)

cc_library(
    name = "collective_aot_test_base",
    testonly = True,
    srcs = ["collective_aot_test_base.cc"],
    hdrs = ["collective_aot_test_base.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla:literal",
        "//xla:xla_proto_cc",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:device_assignment",
        "//xla/service:hlo_module_config",
        "//xla/tests:aot_interception_pjrt_client",
        "//xla/tests/aot_compatibility:test_lib",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

# Multi-GPU collective targets. xla_test appends "_2gpu_<backend>" for the
# "multi_gpu" tag, and every target's goldens live in
# executables/<target name>/v<N>/, so the glob must match that name.
# Coverage is split by runtime thunk family, one target per family.

xla_test(
    name = "data_movement_collectives_aot_test",
    srcs = ["data_movement_collectives_aot_test.cc"],
    backends = _GPU_TARGETS,
    data = glob(["executables/data_movement_collectives_aot_test_2gpu_*/**/*.pbtxt"]),
    disabled_backends = _DISABLED_GPU_TARGETS,
    tags = [
        "multi_gpu",
        "no_oss",
    ],
    deps = [
        ":collective_aot_test_base",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:types",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_module_config",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/aot_compatibility:test_lib",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "p2p_collectives_aot_test",
    srcs = ["p2p_collectives_aot_test.cc"],
    backends = _GPU_TARGETS,
    data = glob(["executables/p2p_collectives_aot_test_2gpu_*/**/*.pbtxt"]),
    disabled_backends = _DISABLED_GPU_TARGETS,
    tags = [
        "multi_gpu",
        "no_oss",
    ],
    deps = [
        ":collective_aot_test_base",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:types",
        "//xla:xla_proto_cc",
        "//xla/service:device_assignment",
        "//xla/service:hlo_module_config",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/aot_compatibility:test_lib",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "reduction_collectives_aot_test",
    srcs = ["reduction_collectives_aot_test.cc"],
    backends = _GPU_TARGETS,
    data = glob(["executables/reduction_collectives_aot_test_2gpu_*/**/*.pbtxt"]),
    disabled_backends = _DISABLED_GPU_TARGETS,
    tags = [
        "multi_gpu",
        "no_oss",
    ],
    deps = [
        ":collective_aot_test_base",
        "//xla:literal",
        "//xla:types",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_module_config",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/aot_compatibility:test_lib",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "ragged_all_to_all_aot_test",
    srcs = ["ragged_all_to_all_aot_test.cc"],
    backends = _GPU_TARGETS,
    data = glob(["executables/ragged_all_to_all_aot_test_2gpu_*/**/*.pbtxt"]),
    disabled_backends = _DISABLED_GPU_TARGETS,
    tags = [
        "multi_gpu",
        "no_oss",
    ],
    deps = [
        ":collective_aot_test_base",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_module_config",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/aot_compatibility:test_lib",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "collective_kernels_aot_test",
    srcs = ["collective_kernels_aot_test.cc"],
    backends = _GPU_TARGETS,
    data = glob(["executables/collective_kernels_aot_test_2gpu_*/**/*.pbtxt"]),
    disabled_backends = _DISABLED_GPU_TARGETS,
    tags = [
        "multi_gpu",
        "no_oss",
    ],
    deps = [
        ":collective_aot_test_base",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:types",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_module_config",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/aot_compatibility:test_lib",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "async_collectives_aot_test",
    srcs = ["async_collectives_aot_test.cc"],
    backends = _GPU_TARGETS,
    data = glob(["executables/async_collectives_aot_test_2gpu_*/**/*.pbtxt"]),
    disabled_backends = _DISABLED_GPU_TARGETS,
    tags = [
        "multi_gpu",
        "no_oss",
    ],
    deps = [
        ":collective_aot_test_base",
        "//xla:literal",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_module_config",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/aot_compatibility:test_lib",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# LEGACY: the original collective target, kept only for its v1-v3 goldens from
# older compilers. Do not add test cases to it; it is scheduled for deletion
# (see the TODO in collective_ops_aot_test.cc).
xla_test(
    name = "collective_ops_aot_test",
    srcs = ["collective_ops_aot_test.cc"],
    backends = _GPU_TARGETS,
    data = glob(["executables/collective_ops_aot_test_2gpu_*/**/*.pbtxt"]),
    disabled_backends = _DISABLED_GPU_TARGETS,
    tags = [
        "multi_gpu",
        "no_oss",
    ],
    deps = [
        ":collective_aot_test_base",
        "//xla:error_spec",
        "//xla:literal",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/aot_compatibility:test_lib",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
    ],
)
