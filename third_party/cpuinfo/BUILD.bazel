# cpuinfo, a library to detect information about the host CPU
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

C99OPTS = [
    "-std=gnu99",  # gnu99, not c99, because dprintf is used
    "-Wno-vla",
    "-D_GNU_SOURCE=1",  # to use CPU_SETSIZE
    "-DCPUINFO_INTERNAL=",
    "-DCPUINFO_PRIVATE=",
]

# Source code common to all platforms.
COMMON_SRCS = [
    "src/api.c",
    "src/init.c",
    "src/cache.c",
]

# Architecture-specific sources and headers.
X86_SRCS = [
    "src/x86/cache/descriptor.c",
    "src/x86/cache/deterministic.c",
    "src/x86/cache/init.c",
    "src/x86/info.c",
    "src/x86/init.c",
    "src/x86/isa.c",
    "src/x86/name.c",
    "src/x86/topology.c",
    "src/x86/uarch.c",
    "src/x86/vendor.c",
]

ARM_SRCS = [
    "src/arm/cache.c",
    "src/arm/uarch.c",
]

# Platform-specific sources and headers
LINUX_SRCS = [
    "src/linux/cpulist.c",
    "src/linux/multiline.c",
    "src/linux/processors.c",
    "src/linux/smallfile.c",
]

MOCK_LINUX_SRCS = [
    "src/linux/mockfile.c",
]

MACH_SRCS = [
    "src/mach/topology.c",
]

EMSCRIPTEN_SRCS = [
    "src/emscripten/init.c",
]

LINUX_X86_SRCS = [
    "src/x86/linux/cpuinfo.c",
    "src/x86/linux/init.c",
]

LINUX_ARM_SRCS = [
    "src/arm/linux/chipset.c",
    "src/arm/linux/clusters.c",
    "src/arm/linux/cpuinfo.c",
    "src/arm/linux/hwcap.c",
    "src/arm/linux/init.c",
    "src/arm/linux/midr.c",
]

LINUX_ARM32_SRCS = LINUX_ARM_SRCS + ["src/arm/linux/aarch32-isa.c"]

LINUX_ARM64_SRCS = LINUX_ARM_SRCS + ["src/arm/linux/aarch64-isa.c"]

ANDROID_ARM_SRCS = [
    "src/arm/android/properties.c",
]

WINDOWS_X86_SRCS = [
    "src/x86/windows/init.c",
]

MACH_X86_SRCS = [
    "src/x86/mach/init.c",
]

MACH_ARM_SRCS = [
    "src/arm/mach/init.c",
]

cc_library(
    name = "cpuinfo_impl",
    srcs = select({
        ":linux_x86_64": COMMON_SRCS + X86_SRCS + LINUX_SRCS + LINUX_X86_SRCS,
        ":linux_arm": COMMON_SRCS + ARM_SRCS + LINUX_SRCS + LINUX_ARM32_SRCS,
        ":linux_armhf": COMMON_SRCS + ARM_SRCS + LINUX_SRCS + LINUX_ARM32_SRCS,
        ":linux_aarch64": COMMON_SRCS + ARM_SRCS + LINUX_SRCS + LINUX_ARM64_SRCS,
        ":macos_x86_64": COMMON_SRCS + X86_SRCS + MACH_SRCS + MACH_X86_SRCS,
        ":windows_x86_64": COMMON_SRCS + X86_SRCS + WINDOWS_X86_SRCS,
        ":android_armv7": COMMON_SRCS + ARM_SRCS + LINUX_SRCS + LINUX_ARM32_SRCS + ANDROID_ARM_SRCS,
        ":android_arm64": COMMON_SRCS + ARM_SRCS + LINUX_SRCS + LINUX_ARM64_SRCS + ANDROID_ARM_SRCS,
        ":android_x86": COMMON_SRCS + X86_SRCS + LINUX_SRCS + LINUX_X86_SRCS,
        ":android_x86_64": COMMON_SRCS + X86_SRCS + LINUX_SRCS + LINUX_X86_SRCS,
        ":ios_x86_64": COMMON_SRCS + X86_SRCS + MACH_SRCS + MACH_X86_SRCS,
        ":ios_x86": COMMON_SRCS + X86_SRCS + MACH_SRCS + MACH_X86_SRCS,
        ":ios_armv7": COMMON_SRCS + MACH_SRCS + MACH_ARM_SRCS,
        ":ios_arm64": COMMON_SRCS + MACH_SRCS + MACH_ARM_SRCS,
        ":ios_arm64e": COMMON_SRCS + MACH_SRCS + MACH_ARM_SRCS,
        ":watchos_x86_64": COMMON_SRCS + X86_SRCS + MACH_SRCS + MACH_X86_SRCS,
        ":watchos_x86": COMMON_SRCS + X86_SRCS + MACH_SRCS + MACH_X86_SRCS,
        ":watchos_armv7k": COMMON_SRCS + MACH_SRCS + MACH_ARM_SRCS,
        ":watchos_arm64_32": COMMON_SRCS + MACH_SRCS + MACH_ARM_SRCS,
        ":tvos_x86_64": COMMON_SRCS + X86_SRCS + MACH_SRCS + MACH_X86_SRCS,
        ":tvos_arm64": COMMON_SRCS + MACH_SRCS + MACH_ARM_SRCS,
    }),
    copts = select({
        ":windows_x86_64": [],
        "//conditions:default": C99OPTS,
    }) + [
        "-Iexternal/cpuinfo/include",
        "-Iexternal/cpuinfo/src",
    ],
    linkstatic = True,
    # Headers must be in textual_hdrs to allow us to set the standard to C99
    textual_hdrs = [
        "include/cpuinfo.h",
        "src/linux/api.h",
        "src/mach/api.h",
        "src/cpuinfo/common.h",
        "src/cpuinfo/internal-api.h",
        "src/cpuinfo/log.h",
        "src/cpuinfo/utils.h",
        "src/x86/api.h",
        "src/x86/cpuid.h",
        "src/x86/linux/api.h",
        "src/arm/android/api.h",
        "src/arm/linux/api.h",
        "src/arm/linux/cp.h",
        "src/arm/api.h",
        "src/arm/midr.h",
    ],
    deps = [
        "@clog",
    ],
)

cc_library(
    name = "cpuinfo",
    hdrs = [
        "include/cpuinfo.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":cpuinfo_impl",
    ],
)

############################# Build configurations #############################

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
)

config_setting(
    name = "linux_arm",
    values = {"cpu": "arm"},
)

config_setting(
    name = "linux_armhf",
    values = {"cpu": "armhf"},
)

config_setting(
    name = "linux_aarch64",
    values = {"cpu": "aarch64"},
)

config_setting(
    name = "macos_x86_64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
)

config_setting(
    name = "windows_x86_64",
    values = {"cpu": "x64_windows"},
)

config_setting(
    name = "android_armv7",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi-v7a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "arm64-v8a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86_64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86_64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios_armv7",
    values = {
        "apple_platform_type": "ios",
        "cpu": "ios_armv7",
    },
)

config_setting(
    name = "ios_arm64",
    values = {
        "apple_platform_type": "ios",
        "cpu": "ios_arm64",
    },
)

config_setting(
    name = "ios_arm64e",
    values = {
        "apple_platform_type": "ios",
        "cpu": "ios_arm64e",
    },
)

config_setting(
    name = "ios_x86",
    values = {
        "apple_platform_type": "ios",
        "cpu": "ios_i386",
    },
)

config_setting(
    name = "ios_x86_64",
    values = {
        "apple_platform_type": "ios",
        "cpu": "ios_x86_64",
    },
)

config_setting(
    name = "watchos_armv7k",
    values = {
        "apple_platform_type": "watchos",
        "cpu": "watchos_armv7k",
    },
)

config_setting(
    name = "watchos_arm64_32",
    values = {
        "apple_platform_type": "watchos",
        "cpu": "watchos_arm64_32",
    },
)

config_setting(
    name = "watchos_x86",
    values = {
        "apple_platform_type": "watchos",
        "cpu": "watchos_i386",
    },
)

config_setting(
    name = "watchos_x86_64",
    values = {
        "apple_platform_type": "watchos",
        "cpu": "watchos_x86_64",
    },
)

config_setting(
    name = "tvos_arm64",
    values = {
        "apple_platform_type": "tvos",
        "cpu": "tvos_arm64",
    },
)

config_setting(
    name = "tvos_x86_64",
    values = {
        "apple_platform_type": "tvos",
        "cpu": "tvos_x86_64",
    },
)

config_setting(
    name = "emscripten_wasm",
    values = {
        "cpu": "wasm",
    },
)

config_setting(
    name = "emscripten_wasmsimd",
    values = {
        "cpu": "wasm",
        "features": "wasm_simd",
    },
)

config_setting(
    name = "emscripten_asmjs",
    values = {
        "cpu": "asmjs",
    },
)
