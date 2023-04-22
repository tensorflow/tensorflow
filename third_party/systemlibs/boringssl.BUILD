licenses(["notice"])

filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "crypto",
    linkopts = ["-lcrypto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ssl",
    linkopts = ["-lssl"],
    visibility = ["//visibility:public"],
    deps = [
        ":crypto",
    ],
)
