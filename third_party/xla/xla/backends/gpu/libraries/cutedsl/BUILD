load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")

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

cc_library(
    name = "ffi",
    srcs = [
        "ffi.cc",
        "//xla/ffi/api:all_headers",
    ],
    tags = ["no_oss"],
    deps = [
        "//xla/ffi",
    ] + if_google([
        "//third_party/py/cutlass/runtime",
    ]),
    alwayslink = 1,
)

xla_test(
    name = "ffi_test",
    srcs = ["ffi_test.cc"],
    backends = ["b200"],
    data = ["vector_add.hlo"],
    deps = [
        ":ffi",
        "//xla:error_spec",
        "//xla/backends/gpu/tests:hlo_pjrt_gpu_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
    ],
)
