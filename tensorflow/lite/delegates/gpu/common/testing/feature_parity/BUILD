load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_testonly = 1,
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "feature_parity",
    hdrs = ["feature_parity.h"],
    deps = [
        ":utils",
        "//tensorflow/lite/delegates/gpu/common/testing/feature_parity/generators:all_generators",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "opengl_test",
    srcs = ["opengl_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv3",
    ],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":feature_parity",
        ":utils",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/lite/delegates/gpu:gl_delegate",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "opencl_test",
    srcs = ["opencl_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":feature_parity",
        ":utils",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/lite/delegates/gpu:delegate",
        "//tensorflow/lite/delegates/gpu:delegate_options",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "xnnpack_test",
    srcs = ["xnnpack_test.cc"],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":feature_parity",
        ":utils",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)
