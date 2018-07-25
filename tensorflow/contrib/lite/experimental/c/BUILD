package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/contrib/lite:build_def.bzl",
    "tflite_cc_shared_object",
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
        ":exported_symbols.lds",
        ":version_script.lds",
    ],
)

tflite_jni_binary(
    name = "libtensorflowlite_c_jni.so",
    linkscript = ":version_script.lds",
    deps = [":c_api"],
)

cc_library(
    name = "c_api",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    deps = [
        "//tensorflow/contrib/lite:context",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:schema_fbs_version",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ],
)

cc_test(
    name = "c_api_test",
    size = "small",
    srcs = ["c_api_test.cc"],
    data = ["//tensorflow/contrib/lite:testdata/add.bin"],
    deps = [
        ":c_api",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:kernel_api",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)
