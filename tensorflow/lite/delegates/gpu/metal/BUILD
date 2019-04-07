package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

DEFAULT_COPTS = [
    "-std=c++11",
    "-Wno-shorten-64-to-32",
]

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    deps = [
        ":compiled_model",
        ":compute_task_descriptor",
        ":environment",
        ":runtime_options",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/metal/kernels",
    ],
)

objc_library(
    name = "buffer_convert",
    srcs = ["buffer_convert.mm"],
    hdrs = ["buffer_convert.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = [
        "Metal",
    ],
    deps = [
        ":common",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:util",
    ],
)

objc_library(
    name = "common",
    srcs = ["common.mm"],
    hdrs = ["common.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = [
        "Metal",
        "UIKit",
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "compiled_model",
    srcs = ["compiled_model.cc"],
    hdrs = ["compiled_model.h"],
    deps = [
        ":compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "compute_task",
    srcs = ["compute_task.mm"],
    hdrs = ["compute_task.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = ["Metal"],
    deps = [
        ":common",
        ":compute_task_descriptor",
        ":runtime_options",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
    ],
)

objc_library(
    name = "compute_task_descriptor",
    srcs = ["compute_task_descriptor.cc"],
    hdrs = ["compute_task_descriptor.h"],
    copts = DEFAULT_COPTS,
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@FP16",
    ],
)

objc_library(
    name = "environment",
    srcs = ["environment.mm"],
    hdrs = ["environment.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = [
        "Metal",
        "UIKit",
    ],
    deps = [
        ":common",
    ],
)

objc_library(
    name = "inference_context",
    srcs = ["inference_context.mm"],
    hdrs = ["inference_context.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = ["Metal"],
    deps = [
        ":compute_task",
        ":compute_task_descriptor",
        ":runtime_options",
        "//tensorflow/lite/delegates/gpu/common:memory_management",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "runtime_options",
    hdrs = ["runtime_options.h"],
)
