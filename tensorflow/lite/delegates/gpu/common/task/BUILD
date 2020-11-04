load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "buffer_desc",
    srcs = ["buffer_desc.cc"],
    hdrs = ["buffer_desc.h"],
    deps = [
        ":gpu_object_desc",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_object_desc",
    hdrs = ["gpu_object_desc.h"],
    deps = [
        ":serialization_base_cc_fbs",
        "//tensorflow/lite/delegates/gpu/common:access_type",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

flatbuffer_cc_library(
    name = "serialization_base_cc_fbs",
    srcs = ["serialization_base.fbs"],
    flatc_args = [
        "--scoped-enums",
    ],
)

cc_library(
    name = "texture2d_desc",
    srcs = ["texture2d_desc.cc"],
    hdrs = ["texture2d_desc.h"],
    deps = [
        ":gpu_object_desc",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        ":gpu_object_desc",
    ],
)
