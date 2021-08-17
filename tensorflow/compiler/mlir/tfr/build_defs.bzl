"""BUILD extension for TF composition project."""

load("//tensorflow:tensorflow.bzl", "py_binary", "tf_custom_op_library", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

def gen_op_libraries(
        name,
        src,
        deps = [],
        tags = [],
        test = False):
    """gen_op_libraries() generates all cc and py libraries for composite op source.

    Args:
        name: used as the name component of all the generated libraries.
        src: File contains the composite ops.
        deps: Libraries the 'src' depends on.
        tags:
        test:
    """
    if not src.endswith(".py") or name == src[:-3]:
        fail("'src' %s conflicts with op Python wrapper. Rename it to be different from 'name'." % src)

    gen_op_lib_exec = src[:-3]  # Strip off the .py
    py_binary(
        name = gen_op_lib_exec,
        srcs = [src],
        srcs_version = "PY3",
        python_version = "PY3",
        deps = [
            "//tensorflow/compiler/mlir/tfr:op_reg_gen",
            "//tensorflow/compiler/mlir/tfr:tfr_gen",
            "//tensorflow/compiler/mlir/tfr:composite",
        ] + deps,
    )

    registered_op = "registered_" + name
    native.genrule(
        name = registered_op,
        srcs = [],
        outs = [name + ".inc.cc"],
        cmd = "$(location %s) --output=$@ --gen_register_op=true" % gen_op_lib_exec,
        tools = [":" + gen_op_lib_exec],
        tags = tags,
    )

    native.cc_library(
        name = name + "_cc",
        testonly = test,
        srcs = [":" + registered_op],
        deps = [
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
        ],
        alwayslink = 1,
    )

    tf_custom_op_library(
        name = name + ".so",
        srcs = [":" + registered_op],
    )

    tf_gen_op_wrapper_py(
        name = "gen_" + name,
        out = "gen_" + name + ".py",
        deps = [
            ":%s_cc" % name,
        ],
    )

    tf_custom_op_py_library(
        name = name,
        dso = [":%s.so" % name],
        kernels = [":%s_cc" % name],
        srcs_version = "PY3",
        deps = [
            ":gen_%s" % name,
        ],
    )

    # Link the register op and rebuild the binary
    gen_tfr_lib_exec = gen_op_lib_exec + "_with_op_library"
    py_binary(
        name = gen_tfr_lib_exec,
        main = src,
        srcs = [src],
        python_version = "PY3",
        srcs_version = "PY3",
        deps = [
            "//tensorflow/compiler/mlir/tfr:op_reg_gen",
            "//tensorflow/compiler/mlir/tfr:tfr_gen",
            "//tensorflow/compiler/mlir/tfr:composite",
            ":%s" % name,
        ] + deps,
    )

    native.genrule(
        name = name + "_mlir",
        srcs = [],
        outs = [name + ".mlir"],
        cmd = "$(location %s) --output=$@ --gen_register_op=false" % gen_tfr_lib_exec,
        tools = [":" + gen_tfr_lib_exec],
        tags = tags,
    )

    native.py_library(
        name = name + "_py",
        srcs = [src],
        srcs_version = "PY3",
        deps = [
            "//tensorflow/compiler/mlir/tfr:op_reg_gen",
            "//tensorflow/compiler/mlir/tfr:tfr_gen",
            "//tensorflow/compiler/mlir/tfr:composite",
        ] + deps,
    )

def gen_op_bindings(name):
    native.cc_library(
        name = name + "_ops_cc",
        srcs = [name + "_ops.cc"],
        deps = [
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
        ],
        alwayslink = 1,
    )

    tf_custom_op_library(
        name = name + "_ops.so",
        srcs = [name + "_ops.cc"],
    )

    tf_gen_op_wrapper_py(
        name = "gen_" + name + "_ops",
        out = "gen_" + name + "_ops.py",
        deps = [
            ":" + name + "_ops_cc",
        ],
    )

    tf_custom_op_py_library(
        name = name + "_ops",
        dso = [":" + name + "_ops.so"],
        kernels = [":" + name + "_ops_cc"],
        visibility = ["//visibility:public"],
        deps = [
            ":gen_" + name + "_ops",
        ],
    )
