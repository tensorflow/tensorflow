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
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "gpu_operation",
    srcs = ["gpu_operation.cc"],
    hdrs = ["gpu_operation.h"],
    deps = [
        ":serialization_base_cc_fbs",
        "//tensorflow/lite/delegates/gpu/common:access_type",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:kernel_info",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:arguments",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:compiler_options",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_tensor",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tuning_type",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_tensor",
    hdrs = ["gpu_tensor.h"],
)

cc_library(
    name = "profiling_info",
    srcs = ["profiling_info.cc"],
    hdrs = ["profiling_info.h"],
    deps = [
        "@com_google_absl//absl/time",
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
    name = "storage_type_util",
    srcs = ["storage_type_util.cc"],
    hdrs = ["storage_type_util.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "@com_google_absl//absl/strings",
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
    name = "testing_util",
    testonly = 1,
    srcs = ["testing_util.cc"],
    hdrs = ["testing_util.h"],
    deps = [
        ":gpu_operation",
        ":tensor_desc",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:tensor",
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
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "weights_conversion",
    srcs = ["weights_conversion.cc"],
    hdrs = ["weights_conversion.h"],
    deps = [
        ":weights_layout",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "weights_layout",
    srcs = ["weights_layout.cc"],
    hdrs = ["weights_layout.h"],
)

cc_library(
    name = "work_group_picking",
    srcs = ["work_group_picking.cc"],
    hdrs = ["work_group_picking.h"],
    deps = [
        ":tuning_type",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:kernel_info",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common:workgroup_selection",
    ],
)
