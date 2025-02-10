licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nanobind",
    srcs = glob(
        [
            "src/*.cpp",
        ],
        exclude = ["src/nb_combined.cpp"],
    ),
    copts = ["-fexceptions"],
    defines = select({
        "@rules_python//python/config_settings:is_py_freethreaded": [
            "NB_FREE_THREADED=1",
            "NB_BUILD=1",
            "NB_SHARED=1",
        ],
        "//conditions:default": [
            "NB_BUILD=1",
            "NB_SHARED=1",
        ],
    }),
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
