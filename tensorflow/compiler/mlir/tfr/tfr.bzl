"""BUILD extension for TF composition project."""

load("//tensorflow:tensorflow.bzl", "py_binary", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.google.bzl", "pytype_library")

def gen_op_libraries(
        name,
        src,
        deps,
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

    gen_op_lib_exec = src[:-3]
    py_binary(
        name = gen_op_lib_exec,
        srcs = [src],
        srcs_version = "PY2AND3",
        python_version = "PY3",
        deps = [
            "//tensorflow/python:platform",
        ] + deps,
    )

    register_op = "register_" + name
    native.genrule(
        name = register_op,
        srcs = [],
        outs = [name + ".inc.cc"],
        cmd = "$(location %s) --output=$@ --gen_register_op=true" % gen_op_lib_exec,
        exec_tools = [":" + gen_op_lib_exec],
        local = 1,
        tags = tags,
    )

    native.cc_library(
        name = name + "_cc",
        testonly = test,
        srcs = [":" + register_op],
        copts = [
            "-Wno-unused-result",
            "-Wno-unused-variable",
        ],
        deps = [
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
        ],
        alwayslink = 1,
    )

    tf_gen_op_wrapper_py(
        name = name,
        out = name + ".py",
        deps = [
            ":%s_cc" % name,
        ],
    )

    pytype_library(
        name = name + "_grads",
        srcs = [
            src,
        ],
        srcs_version = "PY2AND3",
        deps = [
            "//third_party/py/numpy",
            "//third_party/py/tensorflow",
        ] + deps,
    )

    pytype_library(
        name = name + "_lib",
        srcs = [
            name + ".py",
        ],
        srcs_version = "PY2AND3",
        deps = [
            ":%s" % name,
            ":%s_cc" % name,
            ":%s_grads" % name,
            "//third_party/py/numpy",
            "//third_party/py/tensorflow",
        ] + deps,
    )

    # Link the register op and rebuild the binary
    gen_tfr_lib_exec = gen_op_lib_exec + "_registered"
    py_binary(
        name = gen_tfr_lib_exec,
        main = src,
        srcs = [src],
        srcs_version = "PY2AND3",
        python_version = "PY3",
        deps = [
            "//tensorflow/python:platform",
            ":%s" % name + "_cc",
        ] + deps,
    )

    op_tfr = "composite_" + name
    native.genrule(
        name = op_tfr,
        srcs = [],
        outs = [name + ".mlir"],
        cmd = "$(location %s) --output=$@ --gen_register_op=false" % gen_tfr_lib_exec,
        exec_tools = [":" + gen_tfr_lib_exec],
        local = 1,
        tags = tags,
    )
