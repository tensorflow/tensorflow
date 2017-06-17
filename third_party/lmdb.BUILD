# Description:
#   LMDB is the Lightning Memory-mapped Database.

licenses(["notice"])  # OpenLDAP Public License

exports_files(["LICENSE"])

cc_library(
    name = "lmdb",
    srcs = [
        "mdb.c",
        "midl.c",
    ],
    hdrs = [
        "lmdb.h",
        "midl.h",
    ],
    copts = [
        "-w",
    ],
    linkopts = select({
        ":windows": ["-Wl,advapi32.lib"],  # InitializeSecurityDescriptor, SetSecurityDescriptorDacl
        ":windows_msvc": ["-Wl,advapi32.lib"],
        "//conditions:default": ["-lpthread"],
    }),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

config_setting(
    name = "windows_msvc",
    values = {"cpu": "x64_windows_msvc"},
)
