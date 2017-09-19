exports_files(["LICENSE"])

cc_library(
    name = "mkl_dnn",
    srcs = glob([
        "src/common/*.cpp",
        "src/cpu/*.cpp",
    ]),
    hdrs = glob(["include/*"]),
    copts = [
        "-fopenmp",
        "-fexceptions",
    ],
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
