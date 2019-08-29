package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bfloat16",
    srcs = ["bfloat16.cc"],
    hdrs = ["bfloat16.h"],
    deps = [
        "//tensorflow/core/platform:byte_order",
        "//third_party/eigen3",
    ],
)

# TODO(bmzhao): Remove the following once references in core/BUILD is removed.
exports_files(
    glob(["*"]),
)
