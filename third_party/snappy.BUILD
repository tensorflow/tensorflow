package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD 3-Clause

exports_files(["COPYING"])

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_msvc",
    values = {"cpu": "x64_windows_msvc"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "snappy",
    srcs = [
        "snappy.cc",
        "snappy.h",
        "snappy-c.cc",
        "snappy-c.h",
        "snappy-internal.h",
        "snappy-sinksource.cc",
        "snappy-sinksource.h",
        "snappy-stubs-internal.cc",
        "snappy-stubs-internal.h",
        "snappy-stubs-public.h",
    ],
    hdrs = ["snappy.h"],
    copts = select({
        ":windows": [],
        ":windows_msvc": [],
        "//conditions:default": [
            "-Wno-shift-negative-value",
            "-Wno-implicit-function-declaration",
        ],
    }),
)

genrule(
    name = "snappy_stubs_public_h",
    srcs = ["snappy-stubs-public.h.in"],
    outs = ["snappy-stubs-public.h"],
    cmd = ("sed " +
           "-e 's/@ac_cv_have_stdint_h@/1/g' " +
           "-e 's/@ac_cv_have_stddef_h@/1/g' " +
           "-e 's/@ac_cv_have_stdint_h@/1/g' " +
           select({
               "@org_tensorflow//tensorflow:windows": "-e 's/@ac_cv_have_sys_uio_h@/0/g' ",
               "@org_tensorflow//tensorflow:windows_msvc": "-e 's/@ac_cv_have_sys_uio_h@/0/g' ",
               "//conditions:default": "-e 's/@ac_cv_have_sys_uio_h@/1/g' ",
           }) +
           "-e 's/@SNAPPY_MAJOR@/1/g' " +
           "-e 's/@SNAPPY_MINOR@/1/g' " +
           "-e 's/@SNAPPY_PATCHLEVEL@/4/g' " +
           "$< >$@"),
)
