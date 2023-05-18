load("//tensorflow/compiler/xla/tests/fuzz:build_defs.bzl", "hlo_test")

hlo_test(
    name = "rand",
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
)
