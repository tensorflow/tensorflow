package(
    default_visibility = ["//tensorflow/c/experimental/ops/gen:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "model",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:str_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)
