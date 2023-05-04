load("//tensorflow/compiler/xla/tests/fuzz:build_defs.bzl", "hlo_test")

hlo_test(
    name = "rand",
    hlo_files = glob(
        include = ["rand_*.hlo"],
        exclude = [
            "rand_1.hlo",  # fails on GPU
            "rand_4.hlo",  # times out during coverage
            # These fail on all platforms
            "rand_60.hlo",
            "rand_67.hlo",
            "rand_72.hlo",
	    "rand_79.hlo",  # No int8 support on ROCM yet
        ],
    ),
)
