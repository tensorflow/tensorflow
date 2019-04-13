DEFAULT_COPTS = ["-Wno-unused-value",
                 "-D_GLIBCXX_USE_CXX11_ABI=0"]

def boost_library(name, defines=None, deps=None, extra_srcs=None, extra_hdrs=None, linkopts=None):
    if defines == None:
        defines = []

    if deps == None:
        deps = []

    if extra_srcs == None:
        extra_srcs = []

    if extra_hdrs == None:
        extra_hdrs = []

    if linkopts == None:
        linkopts = []

    return native.cc_library(
        name = name,
        visibility = ["//visibility:public"],
        defines = defines,
        hdrs = native.glob([
            x % name
            for x in [
                'libs/%s/src/*.hpp',
                'boost/%s/**/*.ipp',
            ]
        ]) + extra_hdrs,
        srcs = native.glob([
            x % name
            for x in [
                'libs/%s/src/*.cpp',
                'boost/%s/detail/*.hpp',
            ]
        ]) + extra_srcs,
        deps = deps,
        copts = DEFAULT_COPTS,
        linkopts = linkopts,
    )
