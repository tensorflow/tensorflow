exports_files(["LICENSE"])

cc_library(
    name = "mkl_dnn",
    srcs = glob([
        "src/common/*.cpp",
        "src/cpu/*.cpp",
    ]),
    hdrs = glob(["include/*"]),
    copts = ["-fexceptions"] + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "-fopenmp",
        ],
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
