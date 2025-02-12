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
    copts = [
        "-fexceptions",
        "-Wno-undefined-inline",
        "-Wno-pragma-once-outside-header",
    ],
    includes = ["include"],
    strip_include_prefix = "include",
    deps = [
        "@local_tsl//third_party/python_runtime:headers",
    ],
)

# Needed by pybind11_bazel.
config_setting(
    name = "osx",
    constraint_values = ["@platforms//os:osx"],
)
