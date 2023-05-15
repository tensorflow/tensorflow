load("//tensorflow/compiler/xla/tests/fuzz:build_defs.bzl", "hlo_test")

hlo_test(
    name = "rand",
    hlo_files = glob(
        include = ["rand_*.hlo"],
        exclude = [
            "rand_000001.hlo",  # fails on GPU
            "rand_000004.hlo",  # times out during coverage
            # These fail on all platforms
<<<<<<< HEAD
            "rand_60.hlo",
            "rand_67.hlo",
            "rand_72.hlo",
	    "rand_79.hlo",  # No int8 support on ROCM yet
=======
            "rand_000060.hlo",
            "rand_000067.hlo",
            "rand_000072.hlo",
>>>>>>> upstream/master
        ],
    ),
)
