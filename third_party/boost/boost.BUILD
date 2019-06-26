package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load("@org_tensorflow//third_party/boost:boost.bzl", "boost_library")

cc_library(
    name = 'headers',
    visibility = ["//visibility:public"],
    # includes: list of include dirs added to the compile line
    includes = [
        ".",
    ],
    hdrs = glob([
        "boost/**/*.h",
        "boost/**/*.hpp",
        "boost/**/*.ipp",
    ]),
)

boost_library(
    name = 'filesystem',
    deps = [
        ':system',
    ]
)

boost_library(
    name = 'program_options',
    deps = [
        ':headers',
    ],
)

boost_library(
    name = 'regex',
    deps = [
        ':headers',
    ],
)

boost_library(
    name = 'system',
    deps = [
        ':headers',
    ],
)

boost_library(
    name = 'thread',
    deps = [
        ':headers',
    ],
    # Add source files for the pthread backend
    extra_srcs = glob([
        "libs/thread/src/pthread/once.cpp",
        "libs/thread/src/pthread/thread.cpp",
    ]),
    extra_hdrs = [
        "libs/thread/src/pthread/once_atomic.cpp",
    ],
    linkopts = [
        '-pthread',
    ],
)
