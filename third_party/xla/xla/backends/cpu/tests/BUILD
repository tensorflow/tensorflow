load("//xla/tests:build_defs.bzl", "xla_test")

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

xla_test(
    name = "gemv_test",
    srcs = ["gemv_test.cc"],
    backends = ["cpu"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/parser:hlo_parser",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/tests:client_library_test_runner_mixin",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "ynn_fusion_test",
    srcs = ["ynn_fusion_test.cc"],
    backends = ["cpu"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:error_spec",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:platform_port",
    ],
)
