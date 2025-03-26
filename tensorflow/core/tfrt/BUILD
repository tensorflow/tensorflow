# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "stream_ops_op_lib",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/tfrt/kernels:stream_ops",
        "//tensorflow/core/tfrt/ops:stream_ops_op_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ifrt_program_ops_op_lib",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/tfrt/kernels:ifrt_program_ops",
        "//tensorflow/core/tfrt/ops:ifrt_program_ops_op_lib",
    ],
    alwayslink = 1,
)
