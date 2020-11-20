load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "arguments",
    srcs = ["arguments.cc"],
    hdrs = ["arguments.h"],
    deps = [
        ":serialization_base_cc_fbs",
        "//tensorflow/lite/delegates/gpu/common:access_type",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_object_desc",
        "@com_google_absl//absl/strings",
    ],
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
    name = "compiler_options",
    hdrs = ["compiler_options.h"],
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

cc_library(
    name = "gpu_tensor",
    hdrs = ["gpu_tensor.h"],
)

flatbuffer_cc_library(
    name = "serialization_base_cc_fbs",
    srcs = ["serialization_base.fbs"],
    flatc_args = [
        "--scoped-enums",
    ],
)

cc_library(
    name = "tensor_desc",
    srcs = ["tensor_desc.cc"],
    hdrs = ["tensor_desc.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_object_desc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tensor_linear_desc",
    srcs = ["tensor_linear_desc.cc"],
    hdrs = ["tensor_linear_desc.h"],
    deps = [
        ":gpu_object_desc",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
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
    name = "tuning_type",
    hdrs = ["tuning_type.h"],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        ":gpu_object_desc",
    ],
)

cc_library(
    name = "weights_layout",
    hdrs = ["weights_layout.h"],
)
