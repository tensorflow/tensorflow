# Description:
#   AWS C++ SDK

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("@%ws%//third_party:common.bzl", "template_rule")

cc_library(
    name = "aws",
    srcs = select({
        ":linux_x86_64": glob([
            "aws-cpp-sdk-core/source/platform/linux-shared/*.cpp",
        ]),
        "//conditions:default": glob([
            "aws-cpp-sdk-core/source/*.cpp",
            "aws-cpp-sdk-core/source/auth/*.cpp",
            "aws-cpp-sdk-core/source/config/*.cpp",
            "aws-cpp-sdk-core/source/client/*.cpp",
            "aws-cpp-sdk-core/source/external/*.cpp",
            "aws-cpp-sdk-core/source/internal/*.cpp",
            "aws-cpp-sdk-core/source/utils/*.cpp",
            "aws-cpp-sdk-s3/source/**/*.cpp",
        ]),
    }),
    includes = [
        "aws-cpp-sdk-core/include/",
        "aws-cpp-sdk-s3/include/",
    ],
    defines = select({
        ":linux_x86_64": ["PLATFORM_LINUX"],
        "//conditions:default": [],
    }),
    hdrs = [
        "aws-cpp-sdk-core/include/aws/core/SDKConfig.h",
    ],
)

template_rule(
    name = "SDKConfig_h",
    src = "aws-cpp-sdk-core/include/aws/core/SDKConfig.h.in",
    out = "aws-cpp-sdk-core/include/aws/core/SDKConfig.h",
    substitutions = {
        "cmakedefine": "define",
    },
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)
