exports_files(["LICENSE"])

config_setting(
    name = "clang_linux_x86_64",
    values = {
        "cpu": "k8",
        "define": "using_clang=true",
    },
)

cc_library(
    name = "mkl_dnn",
    srcs = glob([
        "src/common/*.cpp",
        "src/cpu/*.cpp",
    ]),
    hdrs = glob(["include/*"]),
    copts = ["-fexceptions"] + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "-fopenmp",  # only works with gcc
        ],
        # TODO(ibiryukov): enable openmp with clang by including libomp as a
        # dependency.
        ":clang_linux_x86_64": [],
        "//conditions:default": [],
    }),
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/xbyak",
    ],
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
)
