load("//tensorflow:tensorflow.bzl", "if_google")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

_selectors_package = (
    "//tensorflow/lite/delegates/gpu/metal/selectors/" +
    if_google("google", "default")
)

cc_library(
    name = "default_selector",
    hdrs = ["default_selector.h"],
    deps = [
        ":subgraph",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:status",
        _selectors_package + ":default_selector",
    ],
)

cc_library(
    name = "operation_selector",
    srcs = ["operation_selector.cc"],
    hdrs = ["operation_selector.h"],
    deps = [
        ":default_selector",
        ":subgraph",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
        "//tensorflow/lite/delegates/gpu/common/tasks:add",
        "//tensorflow/lite/delegates/gpu/common/tasks:concat_xy",
        "//tensorflow/lite/delegates/gpu/common/tasks:concat_z",
        "//tensorflow/lite/delegates/gpu/common/tasks:elementwise",
        "//tensorflow/lite/delegates/gpu/common/tasks:lstm",
        "//tensorflow/lite/delegates/gpu/common/tasks:padding",
        "//tensorflow/lite/delegates/gpu/common/tasks:prelu",
        "//tensorflow/lite/delegates/gpu/common/tasks:quantize_and_dequantize",
        "//tensorflow/lite/delegates/gpu/common/tasks:reduce",
        "//tensorflow/lite/delegates/gpu/common/tasks:relu",
        "//tensorflow/lite/delegates/gpu/common/tasks:reshape",
        "//tensorflow/lite/delegates/gpu/common/tasks:reshapex4",
        "//tensorflow/lite/delegates/gpu/common/tasks:softmax",
        "//tensorflow/lite/delegates/gpu/common/tasks:softmax1x1",
        "//tensorflow/lite/delegates/gpu/common/tasks:space_to_depth",
        "//tensorflow/lite/delegates/gpu/common/tasks:strided_slice",
        "//tensorflow/lite/delegates/gpu/common/tasks:transpose",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal/kernels",
    ],
)

cc_library(
    name = "subgraph",
    srcs = ["subgraph.cc"],
    hdrs = ["subgraph.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
    ],
)
