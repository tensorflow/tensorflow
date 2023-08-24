load("//tensorflow/compiler/xla/tests/fuzz:build_defs.bzl", "hlo_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "hlo_test_lib",
    testonly = True,
    srcs = ["hlo_test_lib.cc"],
    deps = [
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:env",
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
        "//tensorflow/tsl/platform:test_main",
    ],
)
