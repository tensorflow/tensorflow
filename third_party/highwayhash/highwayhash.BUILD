# Description:
#   SipHash and HighwayHash: cryptographically-strong pseudorandom functions
package(
    default_visibility = ["//visibility:public"],
    features = ["header_modules"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

config_setting(
    name = "haswell",
    values = {"cpu": "haswell"},
)

config_setting(
    name = "k8",
    values = {"cpu": "k8"},
)

config_setting(
    name = "cpu_ppc",
    values = {"cpu": "ppc"},
)

config_setting(
    name = "cpu_aarch64",
    values = {"cpu": "aarch64"},
)

#-----------------------------------------------------------------------------
# Platform-specific

cc_library(
    name = "compiler_specific",
    hdrs = ["highwayhash/compiler_specific.h"],
)

cc_library(
    name = "arch_specific",
    srcs = ["highwayhash/arch_specific.cc"],
    hdrs = ["highwayhash/arch_specific.h"],
    deps = [":compiler_specific"],
)

cc_library(
    name = "endianess",
    hdrs = ["highwayhash/endianess.h"],
)

cc_library(
    name = "instruction_sets",
    srcs = ["highwayhash/instruction_sets.cc"],
    hdrs = ["highwayhash/instruction_sets.h"],
    deps = [
        ":arch_specific",
        ":compiler_specific",
    ],
)

cc_library(
    name = "iaca",
    hdrs = ["highwayhash/iaca.h"],
    deps = [":compiler_specific"],
)

cc_library(
    name = "os_specific",
    srcs = ["highwayhash/os_specific.cc"],
    hdrs = ["highwayhash/os_specific.h"],
    deps = [
        ":arch_specific",
        ":compiler_specific",
    ],
)

#-----------------------------------------------------------------------------
# Vectors

cc_library(
    name = "scalar",
    textual_hdrs = ["highwayhash/scalar.h"],
    deps = [
        ":arch_specific",
        ":compiler_specific",
    ],
)

cc_library(
    name = "vector128",
    textual_hdrs = ["highwayhash/vector128.h"],
    deps = [
        ":arch_specific",
        ":compiler_specific",
    ],
)

cc_library(
    name = "vector256",
    textual_hdrs = ["highwayhash/vector256.h"],
    deps = [
        ":arch_specific",
        ":compiler_specific",
    ],
)

#-----------------------------------------------------------------------------
# SipHash

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
        ":endianess",
    ],
)

#-----------------------------------------------------------------------------
# HighwayHash

cc_library(
    name = "hh_types",
    hdrs = ["highwayhash/hh_types.h"],
    deps = [":instruction_sets"],
)

cc_library(
    name = "load3",
    textual_hdrs = ["highwayhash/load3.h"],
    deps = [
        ":arch_specific",
        ":compiler_specific",
        ":endianess",
    ],
)

cc_library(
    name = "hh_avx2",
    srcs = ["highwayhash/hh_avx2.cc"],
    hdrs = ["highwayhash/highwayhash_target.h"],
    copts = select({
        ":k8": ["-mavx2"],
        ":haswell": ["-mavx2"],
        "//conditions:default": ["-DHH_DISABLE_TARGET_SPECIFIC"],
    }),
    textual_hdrs = [
        "highwayhash/hh_avx2.h",
        "highwayhash/highwayhash_target.cc",
        "highwayhash/highwayhash.h",
        "highwayhash/hh_buffer.h",
    ],
    deps = [
        ":arch_specific",
        ":compiler_specific",
        ":hh_types",
        ":iaca",
        ":load3",
        ":vector128",
        ":vector256",
    ],
)

cc_library(
    name = "hh_sse41",
    srcs = ["highwayhash/hh_sse41.cc"],
    hdrs = ["highwayhash/highwayhash_target.h"],
    copts = select({
        ":k8": ["-msse4.1"],
        ":haswell": ["-msse4.1"],
        "//conditions:default": ["-DHH_DISABLE_TARGET_SPECIFIC"],
    }),
    textual_hdrs = [
        "highwayhash/hh_sse41.h",
        "highwayhash/highwayhash_target.cc",
        "highwayhash/highwayhash.h",
        "highwayhash/hh_buffer.h",
    ],
    deps = [
        ":arch_specific",
        ":compiler_specific",
        ":hh_types",
        ":iaca",
        ":load3",
        ":vector128",
    ],
)

cc_library(
    name = "hh_neon",
    srcs = [
        "highwayhash/hh_neon.cc",
        "highwayhash/vector_neon.h",
    ],
    hdrs = ["highwayhash/highwayhash_target.h"],
    copts = select({
        ":cpu_aarch64": [],
        "//conditions:default": ["-DHH_DISABLE_TARGET_SPECIFIC"],
    }),
    textual_hdrs = [
        "highwayhash/highwayhash_target.cc",
        "highwayhash/highwayhash.h",
        "highwayhash/hh_buffer.h",
        "highwayhash/hh_neon.h",
    ],
    deps = [
        ":arch_specific",
        ":compiler_specific",
        ":hh_types",
        ":load3",
    ],
)

cc_library(
    name = "hh_vsx",
    srcs = ["highwayhash/hh_vsx.cc"],
    hdrs = ["highwayhash/highwayhash_target.h"],
    textual_hdrs = [
        "highwayhash/highwayhash_target.cc",
        "highwayhash/highwayhash.h",
        "highwayhash/hh_buffer.h",
        "highwayhash/hh_vsx.h",
    ],
    deps = [
        ":arch_specific",
        ":compiler_specific",
        ":hh_types",
        ":load3",
    ],
)

cc_library(
    name = "hh_portable",
    srcs = ["highwayhash/hh_portable.cc"],
    hdrs = ["highwayhash/highwayhash_target.h"],
    textual_hdrs = [
        "highwayhash/hh_portable.h",
        "highwayhash/highwayhash_target.cc",
        "highwayhash/highwayhash.h",
        "highwayhash/hh_buffer.h",
    ],
    deps = [
        ":arch_specific",
        ":compiler_specific",
        ":hh_types",
        ":iaca",
        ":load3",
        ":scalar",
    ],
)

# For users of the HighwayHashT template
cc_library(
    name = "highwayhash",
    hdrs = ["highwayhash/highwayhash.h"],
    deps = [
        ":arch_specific",
        ":compiler_specific",
        ":hh_portable",
        ":hh_types",
    ] + select({
        ":cpu_ppc": [":hh_vsx"],
        ":cpu_aarch64": [":hh_neon"],
        "//conditions:default": [
            ":hh_avx2",
            ":hh_sse41",
            ":iaca",
        ],
    }),
)

# For users of InstructionSets<HighwayHash> runtime dispatch
cc_library(
    name = "highwayhash_dynamic",
    hdrs = ["highwayhash/highwayhash_target.h"],
    deps = [
        ":arch_specific",
        ":compiler_specific",
        ":hh_portable",
        ":hh_types",
    ] + select({
        ":cpu_ppc": [":hh_vsx"],
        ":cpu_aarch64": [":hh_neon"],
        "//conditions:default": [
            ":hh_avx2",
            ":hh_sse41",
        ],
    }),
)
