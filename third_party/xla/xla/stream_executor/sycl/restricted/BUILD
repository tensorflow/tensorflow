load("//xla:restricted_package.bzl", "xla_restricted_allow", "xla_restricted_verify")
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

xla_test(
    name = "sycl_blas_lt_test",
    srcs = ["sycl_blas_lt_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    use_legacy_runtime = True,
    deps = [
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/stream_executor/sycl:sycl_blas_lt_plugin",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base_legacy",
    ],
)

xla_test(
    name = "sycl_kernel_test",
    srcs = ["sycl_kernel_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    use_legacy_runtime = True,
    deps = [
        "//xla/backends/gpu/runtime:custom_kernel_thunk",
        "//xla/backends/gpu/runtime:thunk_executor",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/sycl:sycl_platform_id",
        "//xla/tests/restricted:llvm_irgen_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_restricted_verify(allowed_targets = [
    xla_restricted_allow("sycl_blas_lt_test"),
    xla_restricted_allow("sycl_kernel_test"),
])
