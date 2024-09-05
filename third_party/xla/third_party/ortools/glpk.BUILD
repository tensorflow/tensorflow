cc_library(
    name = "glpk",
    srcs = glob(
        [
            "glpk-4.52/src/*.c",
            "glpk-4.52/src/*/*.c",
            "glpk-4.52/src/*.h",
            "glpk-4.52/src/*/*.h",
        ],
        exclude = ["glpk-4.52/src/proxy/main.c"],
    ),
    hdrs = [
        "glpk-4.52/src/glpk.h",
    ],
    copts = [
        "-Wno-error",
        "-w",
        "-Iexternal/glpk/glpk-4.52/src",
        "-Iexternal/glpk/glpk-4.52/src/amd",
        "-Iexternal/glpk/glpk-4.52/src/bflib",
        "-Iexternal/glpk/glpk-4.52/src/cglib",
        "-Iexternal/glpk/glpk-4.52/src/colamd",
        "-Iexternal/glpk/glpk-4.52/src/env",
        "-Iexternal/glpk/glpk-4.52/src/minisat",
        "-Iexternal/glpk/glpk-4.52/src/misc",
        "-Iexternal/glpk/glpk-4.52/src/proxy",
        "-Iexternal/glpk/glpk-4.52/src/zlib",
        "-DHAVE_ZLIB",
    ],
    includes = ["glpk-4.52/src"],
    visibility = ["//visibility:public"],
)
