licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "python_headers",
    hdrs = [":python_include"],
    includes = ["python_include"],
)

cc_library(
    name = "numpy_headers",
    hdrs = [":numpy_include"],
    includes = ["numpy_include"],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

%{PYTHON_INCLUDE_GENRULE}

%{NUMPY_INCLUDE_GENRULE}
