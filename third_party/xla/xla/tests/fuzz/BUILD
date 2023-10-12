load("//xla/tests/fuzz:build_defs.bzl", "hlo_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "hlo_test_lib",
    testonly = True,
    srcs = ["hlo_test_lib.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:env",
    ],
)

hlo_test(
    name = "rand",
    srcs = [],
    hlo_files = glob(
        include = ["rand_*.hlo"],
        exclude = [
            "rand_000001.hlo",  # fails on GPU
            "rand_000004.hlo",  # times out during coverage
            # These fail on all platforms
            "rand_000060.hlo",
            "rand_000067.hlo",
            "rand_000072.hlo",
        ],
    ),
    deps = [
        ":hlo_test_lib",
        "@local_tsl//tsl/platform:test_main",
    ],
)
