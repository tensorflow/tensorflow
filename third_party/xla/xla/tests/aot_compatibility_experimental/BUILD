load("@rules_cc//cc:defs.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "test_lib",
    testonly = True,
    srcs = ["test_lib.cc"],
    hdrs = ["test_lib.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/tests:aot_interception_pjrt_client",
        "//xla/tests:hlo_test_base",
        "//xla/tests:pjrt_client_registry",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:path",
    ],
)

xla_cc_test(
    name = "test_lib_test",
    srcs = ["test_lib_test.cc"],
    data = glob(["gpu/executables/test_dummy_test/**"]),
    tags = [
        "manual",
        "no_oss",
    ],
    deps = [
        ":test_lib",
        "//xla:xla_proto_cc",
        "//xla/service/gpu:gpu_executable_proto_cc",
        "//xla/tests:aot_interception_pjrt_client",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@tsl//tsl/platform:path",
    ],
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
    data = glob(["gpu/executables/collective_ops_aot_test_2gpu/**/*.pbtxt"]),
    tags = ["manual"],
    deps = [
        ":test_lib",
        "//xla:literal",
        "//xla/service:hlo_module_config",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:path",
    ],
)
