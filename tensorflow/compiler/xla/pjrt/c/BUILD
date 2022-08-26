load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

cc_library(
    name = "pjrt_c_api_hdrs",
    hdrs = ["pjrt_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_decl",  # TODO(b/238999986): Remove this.
    ],
)

cc_library(
    name = "pjrt_c_api_tpu_hdrs",
    hdrs = ["pjrt_c_api_tpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_wrapper_impl",
    srcs = ["pjrt_c_api_wrapper_impl.cc"],
    hdrs = ["pjrt_c_api_wrapper_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_future",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_conversions",  # TODO(b/238999986): Remove this.
    ],
)

cc_library(
    name = "pjrt_c_api_helpers",
    srcs = ["pjrt_c_api_helpers.cc"],
    hdrs = ["pjrt_c_api_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        "//tensorflow/compiler/xla:status",
    ],
)
