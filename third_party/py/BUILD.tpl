licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# To build Python C/C++ extension on Windows, we need to link to python import library pythonXY.lib
# See https://docs.python.org/3/extending/windows.html
cc_import(
    name = "python_lib",
    interface_library = select({
        ":windows": ":python_import_lib",
        # A placeholder for Unix platforms which makes --no_build happy.
        "//conditions:default": "not-existing.lib",
    }),
    system_provided = 1,
)

cc_library(
    name = "python_headers",
    hdrs = [":python_include"],
    deps = select({
        ":windows": [":python_lib"],
        "//conditions:default": [],
    }),
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
%{PYTHON_IMPORT_LIB_GENRULE}
