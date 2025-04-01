load("//xla/tests/fuzz:build_defs.bzl", "hlo_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "hlo_test_lib",
    testonly = True,
    srcs = ["hlo_test_lib.cc"],
    deps = [
        "//xla:error_spec",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
    ],
    alwayslink = True,  # This library registers test cases at dynamic initialization time.
)

[hlo_test(
    name = hlo + "_test",
    hlo = hlo,
    tags = (
        ["cuda-only"] if hlo == "rand_000079.hlo" else []  # No int8
    ) + [
        "test_migrated_to_hlo_runner_pjrt",
    ],
) for hlo in glob(
    include = ["rand_*.hlo"],
    exclude = [
        "rand_000001.hlo",  # fails on GPU
        "rand_000004.hlo",  # times out during coverage
        # These fail on all platforms
        "rand_000060.hlo",
        "rand_000067.hlo",
        "rand_000072.hlo",
    ],
)]
