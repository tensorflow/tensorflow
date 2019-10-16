load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrappers_cc",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

tf_gen_op_wrappers_cc(
    name = "xrt_ops",
    op_lib_names = [
        "xrt_compile_ops",
        "xrt_state_ops",
        "xrt_execute_op",
    ],
    pkg = "//tensorflow/compiler/xrt",
)
