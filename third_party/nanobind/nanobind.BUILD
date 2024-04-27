licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nanobind",
    srcs = glob([
        "src/*.cpp",
    ]),
    copts = ["-fexceptions"],
    defines = [
        "NB_BUILD=1",
        "NB_SHARED=1",
    ],
    includes = ["include"],
    textual_hdrs = glob(
        [
            "include/**/*.h",
            "src/*.h",
        ],
    ),
    deps = [
        "@local_tsl//third_party/python_runtime:headers",
        "@robin_map",
    ],
)
