load("//xla/tests:build_defs.bzl", "xla_test")

package(
    default_applicable_licenses = ["//tensorflow:license"],
)

xla_test(
    name = "collectives_test",
    srcs = ["collectives_test.cc"],
    args = ["--xla_force_host_platform_device_count=4"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            "no_oss",
        ],
    },
    backends = [
        "gpu",
    ],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla/service:hlo_module_config",
        "//xla/tests:aot_compatibility_test_lib",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:path",
    ],
)
