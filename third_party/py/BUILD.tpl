licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "python_headers",
    hdrs = [":python_include"],
    includes = ["python_include"],
    data = select({
      ":windows" : [":python_import_lib"],
      "//conditions:default": [],
    }),
    linkopts = select({
      # TODO(pcloudy): Ideally, this should just go into deps after resolving
      # https://github.com/bazelbuild/bazel/issues/3237,
      ":windows" : ["$(locations :python_import_lib)"],
      "//conditions:default": [],
    }),
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
%{PYTHON_IMPORT_LIB_GENRULE}
%{NUMPY_INCLUDE_GENRULE}
