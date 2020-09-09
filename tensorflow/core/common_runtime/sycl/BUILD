load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    features = ["-parse_headers"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "sycl_runtime",
    srcs = if_not_windows([
        "sycl_allocator.cc",
        "sycl_device.cc",
        "sycl_device_context.cc",
        "sycl_device_factory.cc",
    ]),
    hdrs = if_not_windows([
        "sycl_allocator.h",
        "sycl_device.h",
        "sycl_util.h",
        "sycl_device_context.h",
    ]),
    copts = tf_copts(),
    linkstatic = 0,
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//third_party/eigen3",
        "@local_config_sycl//sycl",
    ],
    alwayslink = 0,
)
