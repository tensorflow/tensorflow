package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pybind11",
    hdrs = glob(
        include = [
            "include/pybind11/*.h",
            "include/pybind11/detail/*.h",
        ],
        exclude = [
            "include/pybind11/common.h",
            "include/pybind11/eigen.h",
        ],
    ),
    copts = select({
        ":msvc_compiler": [],
        "//conditions:default": [
            "-fexceptions",
            "-Wno-undefined-inline",
            "-Wno-pragma-once-outside-header",
        ],
    }),
    includes = ["include"],
    strip_include_prefix = "include",
    deps = [
        "@xla//third_party/python_runtime:headers",
    ],
)

# Used when one also needs eigen types.
cc_library(
    name = "pybind11_eigen",
    hdrs = glob(
        include = [
            "include/pybind11/*.h",
            "include/pybind11/detail/*.h",
            "include/pybind11/eigen/*.h",
        ],
        exclude = [
            "include/pybind11/common.h",
        ],
    ),
    copts = select({
        ":msvc_compiler": [],
        "//conditions:default": [
            "-fexceptions",
            "-Wno-undefined-inline",
            "-Wno-pragma-once-outside-header",
        ],
    }),
    includes = ["include"],
    strip_include_prefix = "include",
    deps = [
        "@eigen_archive//:eigen3",
        "@xla//third_party/python_runtime:headers",
    ],
)

# Needed by pybind11_bazel.
config_setting(
    name = "msvc_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "msvc-cl"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "osx",
    constraint_values = ["@platforms//os:osx"],
)
