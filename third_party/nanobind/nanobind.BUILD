load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

bool_flag(
    name = "enabled_free_threading",
    build_setting_default = False,
)

config_setting(
    name = "use_enabled_free_threading",
    flag_values = {
        ":enabled_free_threading": "True",
    },
)

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
        ":use_enabled_free_threading": [
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
