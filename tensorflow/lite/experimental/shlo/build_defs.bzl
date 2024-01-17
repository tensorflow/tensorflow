def shlo_visibility():
    return ["//tensorflow/lite/experimental/shlo:__subpackages__"]

def shlo_includes():
    return ["-Ithird_party/tensorflow/lite/experimental/shlo", "-Ithird_party"]

def shlo_cc_library(
        name,
        srcs = [],
        copts = [],
        hdrs = [],
        defines = [],
        includes = [],
        deps = [],
        linkopts = [],
        visibility = shlo_visibility(),
        compatible_with = None,
        testonly = False):
    native.cc_library(
        name = name,
        srcs = srcs,
        copts = shlo_includes() + copts,
        hdrs = hdrs,
        defines = defines,
        includes = includes,
        deps = deps,
        linkopts = linkopts,
        visibility = visibility,
        compatible_with = compatible_with,
        testonly = testonly,
        linkstatic = True,
    )

def shlo_cc_test(
        name,
        srcs = [],
        copts = [],
        data = [],
        deps = []):
    native.cc_test(
        name = name,
        srcs = srcs,
        copts = shlo_includes() + copts,
        data = data,
        deps = deps,
    )

def shlo_cc_binary(
        name,
        srcs = [],
        copts = [],
        data = [],
        deps = []):
    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = shlo_includes() + copts,
        data = data,
        deps = deps,
    )
