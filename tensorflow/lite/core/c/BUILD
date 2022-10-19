load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    licenses = ["notice"],
)

exports_files(
    ["c_api.h"],
)

cc_library(
    name = "c_api",
    hdrs = [
        "c_api.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite/c:c_api_types",
    ],
)

cc_library(
    name = "private_c_api",
    hdrs = [
        "c_api.h",
    ],
    compatible_with = get_compatible_with_portable(),
    # This is a private target, its visibility is set to public only to be
    # used by "tflite_custom_c_library".
    # Do not use this target directly and don't consider it as a part of the public API.
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite/c:c_api_types",
    ],
)
