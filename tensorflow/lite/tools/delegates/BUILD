load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite", "cc_test_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

common_copts = tflite_copts() + tflite_copts_warnings()

cc_library_with_tflite(
    name = "delegate_provider_hdr",
    hdrs = [
        "delegate_provider.h",
    ],
    copts = common_copts,
    tflite_deps = [
        "//tensorflow/lite/c:common",
    ],
    deps = [
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools:tool_params",
    ],
)

cc_library_with_tflite(
    name = "delegate_provider_lib",
    srcs = [
        "delegate_provider.cc",
    ],
    copts = common_copts,
    tflite_deps = [
        ":delegate_provider_hdr",
    ],
)

# A convenient library for all inference execution providers.
cc_library_with_tflite(
    name = "tflite_execution_providers",
    copts = tflite_copts(),
    tflite_deps = [
        ":xnnpack_delegate_provider",
        ":delegate_provider_lib",
    ],
    deps = [
        ":coreml_delegate_provider",
        ":default_execution_provider",
        ":external_delegate_provider",
        ":gpu_delegate_provider",
        ":hexagon_delegate_provider",
        ":nnapi_delegate_provider",
        "//tensorflow/lite/tools/delegates/experimental/stable_delegate:delegate_provider",
    ],
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "default_execution_provider",
    srcs = ["default_execution_provider.cc"],
    copts = tflite_copts(),
    linkstatic = True,
    tflite_deps = [
        ":delegate_provider_hdr",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "gpu_delegate_provider",
    srcs = ["gpu_delegate_provider.cc"],
    copts = common_copts + select({
        "//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "//tensorflow:macos_arm64": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":delegate_provider_hdr",
        "//tensorflow/lite/tools/evaluation:utils",
    ] + select({
        "//tensorflow/lite/delegates/gpu:supports_gpu_delegate": [
            "//tensorflow/lite/delegates/gpu:delegate",
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:ios": [
            "//tensorflow/lite/delegates/gpu:metal_delegate",
        ],
        "//tensorflow:macos_arm64": [
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
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/nnapi:nnapi_util",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
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
        "//tensorflow:arm_any": [
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
        "//tensorflow:macos_arm64": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":delegate_provider_hdr",
        "//tensorflow/lite/tools/evaluation:utils",
    ] + select({
        "//tensorflow:ios": [
            "//tensorflow/lite/delegates/coreml:coreml_delegate",
        ],
        "//tensorflow:macos_arm64": [
            "//tensorflow/lite/delegates/coreml:coreml_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "xnnpack_delegate_provider",
    srcs = ["xnnpack_delegate_provider.cc"],
    copts = tflite_copts(),
    linkstatic = True,
    tflite_deps = [
        ":delegate_provider_hdr",
        "//tensorflow/lite/tools/evaluation:utils",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "//tensorflow/lite/tools:tool_params",
    ],
    alwayslink = 1,
)

cc_test(
    name = "xnnpack_delegate_provider_test",
    srcs = ["xnnpack_delegate_provider_test.cc"],
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_provider_hdr",
        ":xnnpack_delegate_provider",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "//tensorflow/lite/tools:tool_params",
        "@com_google_googletest//:gtest_main",
    ],
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
        "//tensorflow/lite/delegates/external:external_delegate",
    ],
    alwayslink = 1,
)

cc_test_with_tflite(
    name = "delegate_provider_lib_test",
    size = "small",
    srcs = ["delegate_provider_test.cc"],
    copts = tflite_copts(),
    # See details in https://github.com/bazelbuild/bazel/issues/11552 to avoid
    # lazy symbol binding failure on macOS.
    linkstatic = select({
        "//tensorflow:macos": True,
        "//conditions:default": False,
    }),
    tflite_deps = [
        ":default_execution_provider",
        ":delegate_provider_hdr",
        ":delegate_provider_lib",
        ":xnnpack_delegate_provider",
        "//tensorflow/lite/c:test_util",
    ],
    deps = [
        ":nnapi_delegate_provider",
        "//tensorflow/lite/delegates/utils/dummy_delegate:dummy_delegate_provider",
        "//tensorflow/lite/tools:tool_params",
        "@com_google_googletest//:gtest_main",
    ],
)
