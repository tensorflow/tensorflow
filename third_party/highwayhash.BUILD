# Description:
#   SipHash and HighwayHash: cryptographically-strong pseudorandom functions

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "sip_hash",
    srcs = ["highwayhash/sip_hash.cc"],
    hdrs = [
        "highwayhash/sip_hash.h",
        "highwayhash/state_helpers.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":arch_specific",
        ":compiler_specific",
    ],
)

cc_library(
    name = "arch_specific",
    srcs = ["highwayhash/arch_specific.cc"],
    hdrs = ["highwayhash/arch_specific.h"],
    deps = [":compiler_specific"],
)

cc_library(
    name = "compiler_specific",
    hdrs = ["highwayhash/compiler_specific.h"],
)
