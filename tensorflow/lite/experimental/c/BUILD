load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_cc_shared_object",
    "tflite_copts",
)

package(
    default_visibility = [":experimental"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "experimental",
    packages = [
        "//tensorflow/lite/experimental/...",
        "//third_party/dart/tflite_native/...",  # whitelisted
    ],
)

tflite_cc_shared_object(
    name = "libtensorflowlite_c.so",
    linkopts = select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/experimental/c:exported_symbols.lds)",
            "-Wl,-install_name,@rpath/libtensorflowlite_c.so",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //tensorflow/lite/experimental/c:version_script.lds)",
        ],
    }),
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":exported_symbols.lds",
        ":version_script.lds",
    ],
)

cc_library(
    name = "c_api_internal",
    srcs = [
        "c_api.h",
        "c_api_types.h",
    ],
    hdrs = ["c_api_internal.h"],
    copts = tflite_copts(),
    visibility = [
        "//tensorflow/lite/experimental/c:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
    ],
)

cc_library(
    name = "c_api",
    srcs = ["c_api.cc"],
    hdrs = [
        "c_api.h",
        "c_api_types.h",
    ],
    copts = tflite_copts(),
    visibility = [
        ":experimental",
    ],
    deps = [
        ":c_api_internal",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:version",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "c_api_experimental",
    srcs = ["c_api_experimental.cc"],
    hdrs = ["c_api_experimental.h"],
    copts = tflite_copts(),
    deps = [
        ":c_api",
        ":c_api_internal",
        "//tensorflow/lite:kernel_api",
    ],
    alwayslink = 1,
)

cc_test(
    name = "c_api_test",
    size = "small",
    srcs = ["c_api_test.cc"],
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/add_quantized.bin",
    ],
    deps = [
        ":c_api",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "c_api_experimental_test",
    size = "small",
    srcs = ["c_api_experimental_test.cc"],
    data = ["//tensorflow/lite:testdata/add.bin"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)
