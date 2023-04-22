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
        ":windows": ["-DEFAULTLIB:advapi32.lib"],  # InitializeSecurityDescriptor, SetSecurityDescriptorDacl
        "//conditions:default": ["-lpthread"],
    }),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)
