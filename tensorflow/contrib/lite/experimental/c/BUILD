package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/contrib/lite:build_def.bzl",
    "tflite_cc_shared_object",
    "tflite_copts",
    "tflite_jni_binary",
)

tflite_cc_shared_object(
    name = "libtensorflowlite_c.so",
    linkopts = select({
        "//tensorflow:darwin": [
            "-Wl,-exported_symbols_list",  # This line must be directly followed by the exported_symbols.lds file
            "$(location //tensorflow/contrib/lite/experimental/c:exported_symbols.lds)",
            "-Wl,-install_name,@rpath/libtensorflowlite_c.so",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "$(location //tensorflow/contrib/lite/experimental/c:version_script.lds)",
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
    srcs = ["c_api.h"],
    hdrs = ["c_api_internal.h"],
    copts = tflite_copts(),
    visibility = [
        "//tensorflow/contrib/lite/experimental/c:__subpackages__",
    ],
    deps = [
        "//tensorflow/contrib/lite:context",
        "//tensorflow/contrib/lite:framework",
    ],
)

cc_library(
    name = "c_api",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    copts = tflite_copts(),
    deps = [
        ":c_api_internal",
        "//tensorflow/contrib/lite:context",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:schema_fbs_version",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "c_api_experimental",
    srcs = ["c_api_experimental.cc"],
    hdrs = ["c_api_experimental.h"],
    copts = tflite_copts(),
    deps = [
        ":c_api",
        ":c_api_internal",
    ],
)

cc_test(
    name = "c_api_test",
    size = "small",
    srcs = ["c_api_test.cc"],
    data = ["//tensorflow/contrib/lite:testdata/add.bin"],
    deps = [
        ":c_api",
        "//tensorflow/contrib/lite:kernel_api",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "c_api_experimental_test",
    size = "small",
    srcs = ["c_api_experimental_test.cc"],
    data = ["//tensorflow/contrib/lite:testdata/add.bin"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)
