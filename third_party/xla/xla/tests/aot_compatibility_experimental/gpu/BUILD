load("//xla/tests:build_defs.bzl", "xla_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "test_dummy_test_data",
    testonly = True,
    srcs = glob(["executables/test_dummy_test/**"]),
    visibility = ["//xla/tests/aot_compatibility_experimental:__pkg__"],
)

xla_test(
    name = "collective_ops_aot_test",
    srcs = ["collective_ops_aot_test.cc"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            "no_oss",
        ],
    },
    backends = ["gpu"],
    data = glob(["executables/collective_ops_aot_test_2gpu/**/*.pbtxt"]),
    tags = ["manual"],
    deps = [
        "//xla:literal",
        "//xla/service:hlo_module_config",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/aot_compatibility_experimental:test_lib",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:path",
    ],
)
