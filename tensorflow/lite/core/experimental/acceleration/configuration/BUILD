package(
    licenses = ["notice"],
)

cc_library(
    name = "delegate_registry",
    srcs = ["delegate_registry.cc"],
    hdrs = ["delegate_registry.h"],
    visibility = [
        "//tensorflow/lite/experimental/acceleration/configuration:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@com_google_absl//absl/synchronization",
    ],
)
