load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_cc_shared_object",
    "tflite_copts",
    "tflite_custom_c_library",
)
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Generates a platform-specific shared library containing the TensorFlow Lite C
# API implementation as define in `c_api.h`. The exact output library name
# is platform dependent:
#   - Linux/Android: `libtensorflowlite_c.so`
#   - Mac: `libtensorflowlite_c.dylib`
#   - Windows: `tensorflowlite_c.dll`
tflite_cc_shared_object(
    name = "tensorflowlite_c",
    linkopts = select({
        "//tensorflow:ios": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/c:exported_symbols.lds)",
        ],
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/c:exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //tensorflow/lite/c:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":exported_symbols.lds",
        ":version_script.lds",
    ],
)

cc_library(
    name = "c_api_internal",
    hdrs = ["c_api_internal.h"],
    copts = tflite_copts(),
    visibility = ["//visibility:private"],
    deps = [
        ":common",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
    ],
)

cc_library(
    name = "c_api",
    hdrs = ["c_api.h"],
    copts = tflite_copts(),
    deps = [
        ":c_api_types",
        ":c_api_without_op_resolver",
        "//tensorflow/lite:create_op_resolver_with_builtin_ops",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

cc_library(
    name = "c_api_for_testing",
    testonly = True,
    srcs = ["c_api_for_testing.cc"],
    hdrs = ["c_api_for_testing.h"],
    deps = [
        ":c_api",
        ":c_api_internal",
    ],
)

cc_library(
    name = "c_api_without_op_resolver",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    copts = tflite_copts(),
    deps = [
        ":c_api_internal",
        ":c_api_types",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:create_op_resolver_header",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:version",
        "//tensorflow/lite/delegates:interpreter_utils",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

cc_library(
    name = "c_api_experimental",
    srcs = ["c_api_experimental.cc"],
    hdrs = ["c_api_experimental.h"],
    copts = tflite_copts(),
    deps = [
        ":c_api",
        ":c_api_internal",
        ":common",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

cc_library(
    name = "c_api_types",
    hdrs = ["c_api_types.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

cc_test(
    name = "c_api_test",
    size = "small",
    srcs = ["c_api_test.cc"],
    copts = tflite_copts(),
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/add_quantized.bin",
    ],
    deps = [
        ":c_api",
        ":common",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_custom_c_library(
    name = "selectively_built_c_api_test_lib",
    testonly = 1,
    models = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/add_quantized.bin",
    ],
    visibility = ["//visibility:private"],
)

cc_test(
    name = "selectively_built_c_api_test",
    size = "small",
    srcs = ["c_api_test.cc"],
    copts = tflite_copts(),
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/add_quantized.bin",
    ],
    deps = [
        ":common",
        ":selectively_built_c_api_test_lib",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "c_api_experimental_test",
    size = "small",
    srcs = ["c_api_experimental_test.cc"],
    copts = tflite_copts(),
    data = ["//tensorflow/lite:testdata/add.bin"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":common",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/delegates:delegate_test_util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = [
        "builtin_op_data.h",
        "common.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":c_api_types",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

# For use with library targets that can't use relative paths.
exports_files([
    "c_api.h",
    "c_api_experimental.h",
    "c_api_types.h",
    "common.h",
])

# For use in selective build rule for C API.
filegroup(
    name = "c_api_srcs",
    srcs = [
        "c_api.cc",
        "c_api_internal.h",
    ],
)

# Test the C extension API code.
cc_test(
    name = "common_test",
    size = "small",
    srcs = ["common_test.cc"],
    deps = [
        ":c_api_types",
        ":common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "builtin_op_data_test",
    size = "small",
    srcs = ["builtin_op_data_test.cc"],
    copts = ["-Wno-unused-variable"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "c_test",
    size = "small",
    srcs = ["c_test.c"],
    copts = tflite_copts(),
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/multi_signatures.bin",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":common",
    ],
)
