licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "python_headers",
    hdrs = select({
        "windows" : [
            "python_include_windows",
        ],
        "//conditions:default" : [
            "python_include",
        ],
    }),
    includes = select({
        "windows" : [
            "python_include_windows",
        ],
        "//conditions:default" : [
            "python_include",
        ],
    }),
)

cc_library(
    name = "numpy_headers",
    hdrs = select({
        "windows" : [
            "numpy_include_windows",
        ],
        "//conditions:default" : [
            "numpy_include",
        ],
    }),
    includes = select({
        "windows" : [
            "numpy_include_windows",
        ],
        "//conditions:default" : [
            "numpy_include",
        ],
    }),
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

%{PYTHON_INCLUDE_GENRULE}

%{NUMPY_INCLUDE_GENRULE}
