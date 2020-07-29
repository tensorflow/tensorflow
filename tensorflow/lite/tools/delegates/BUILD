load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

common_copts = ["-Wall"] + tflite_copts()

cc_library(
    name = "delegate_provider_hdr",
    hdrs = [
        "delegate_provider.h",
    ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools:tool_params",
    ],
)

# A convenient library for all inference execution providers.
cc_library(
    name = "tflite_execution_providers",
    copts = tflite_copts(),
    deps = [
        ":coreml_delegate_provider",
        ":default_execution_provider",
        ":external_delegate_provider",
        ":gpu_delegate_provider",
        ":hexagon_delegate_provider",
        ":nnapi_delegate_provider",
        ":xnnpack_delegate_provider",
    ],
    alwayslink = 1,
)

cc_library(
    name = "default_execution_provider",
    srcs = ["default_execution_provider.cc"],
    copts = tflite_copts(),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_provider_hdr",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gpu_delegate_provider",
    srcs = ["gpu_delegate_provider.cc"],
    copts = common_copts + select({
        "//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":delegate_provider_hdr",
        "//tensorflow/lite/tools/evaluation:utils",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu:delegate",
        ],
        "//tensorflow:ios": [
            "//tensorflow/lite/delegates/gpu:metal_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "nnapi_delegate_provider",
    srcs = ["nnapi_delegate_provider.cc"],
    copts = common_copts,
    deps = [
        ":delegate_provider_hdr",
        "//tensorflow/lite/tools/evaluation:utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hexagon_delegate_provider",
    srcs = ["hexagon_delegate_provider.cc"],
    copts = common_copts,
    deps = [
        ":delegate_provider_hdr",
        "//tensorflow/lite/tools/evaluation:utils",
    ] + select({
        "//tensorflow:android_arm": [
            "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//tensorflow:android_arm64": [
            "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "coreml_delegate_provider",
    srcs = ["coreml_delegate_provider.cc"],
    copts = common_copts + select({
        "//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":delegate_provider_hdr",
        "//tensorflow/lite/tools/evaluation:utils",
    ] + select({
        "//tensorflow:ios": [
            "//tensorflow/lite/experimental/delegates/coreml:coreml_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "xnnpack_delegate_provider",
    srcs = ["xnnpack_delegate_provider.cc"],
    copts = tflite_copts(),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_provider_hdr",
        "//tensorflow/lite/tools/evaluation:utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "external_delegate_provider",
    srcs = ["external_delegate_provider.cc"],
    copts = tflite_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_provider_hdr",
    ],
    alwayslink = 1,
)
