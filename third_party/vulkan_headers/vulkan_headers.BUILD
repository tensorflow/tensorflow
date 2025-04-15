package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

VULKAN_HDRS = [
    "include/vulkan/vk_platform.h",
    "include/vulkan/vk_sdk_platform.h",
    "include/vulkan/vulkan.h",
    "include/vulkan/vulkan_core.h",
]

VULKAN_TEXTUAL_HDRS = [
    "include/vulkan/vulkan_android.h",
    "include/vulkan/vulkan_fuchsia.h",
    "include/vulkan/vulkan_ggp.h",
    "include/vulkan/vulkan_ios.h",
    "include/vulkan/vulkan_macos.h",
    "include/vulkan/vulkan_metal.h",
    "include/vulkan/vulkan_vi.h",
    "include/vulkan/vulkan_wayland.h",
    "include/vulkan/vulkan_win32.h",
    "include/vulkan/vulkan_xcb.h",
    "include/vulkan/vulkan_xlib.h",
    "include/vulkan/vulkan_xlib_xrandr.h",
]

# The main vulkan public headers for applications. This excludes headers
# designed for ICDs and layers.
cc_library(
    name = "vulkan_headers",
    hdrs = VULKAN_HDRS,
    includes = ["include"],
    textual_hdrs = VULKAN_TEXTUAL_HDRS,
)

# Like :vulkan_headers but defining VK_NO_PROTOTYPES to disable the
# inclusion of C function prototypes. Useful if dynamically loading
# all symbols via dlopen/etc.
cc_library(
    name = "vulkan_headers_no_prototypes",
    hdrs = VULKAN_HDRS,
    defines = ["VK_NO_PROTOTYPES"],
    includes = ["include"],
    textual_hdrs = VULKAN_TEXTUAL_HDRS,
)

# Provides a C++-ish interface to Vulkan. A rational set of defines are also
# set and transitively applied to any callers, as well as providing the
# necessary storage as the set of defines leaves symbols undefined otherwise.
cc_library(
    name = "vulkan_hpp",
    srcs =
        select({
            "@local_xla//xla/tsl:macos": [],
            "@local_xla//xla/tsl:ios": [],
            "//conditions:default": ["tensorflow/vulkan_hpp_dispatch_loader_dynamic.cc"],
        }),
    hdrs = ["include/vulkan/vulkan.hpp"],
    defines = [
        "VULKAN_HPP_ASSERT=",
        "VULKAN_HPP_DISABLE_IMPLICIT_RESULT_VALUE_CAST",
        "VULKAN_HPP_NO_EXCEPTIONS",
        "VULKAN_HPP_TYPESAFE_CONVERSION",
        "VULKAN_HPP_TYPESAFE_EXPLICIT",
    ] + select({
        "@local_xla//xla/tsl:macos": [],
        "@local_xla//xla/tsl:ios": [],
        "//conditions:default": ["VULKAN_HPP_DISPATCH_LOADER_DYNAMIC"],
    }),
    includes = ["include"],
    deps = [":vulkan_headers"],
)
