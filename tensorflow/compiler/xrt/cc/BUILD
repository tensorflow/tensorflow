load("//tensorflow:tensorflow.default.bzl", "tf_gen_op_wrappers_cc")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
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
