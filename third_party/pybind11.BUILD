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
        "-Xclang-only=-Wno-undefined-inline",
        "-Xclang-only=-Wno-pragma-once-outside-header",
        "-Xgcc-only=-Wno-error",  # no way to just disable the pragma-once warning in gcc
    ],
    includes = ["include"],
    deps = [
        "@org_tensorflow//third_party/python_runtime:headers",
    ],
)
