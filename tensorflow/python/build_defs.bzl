load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

# Intended only for use within this directory.
# Generated python wrappers are "private" visibility, users should depend on the
# full python code that incorporates the wrappers.  The generated targets have
# a _gen suffix, so that the full python version can use the bare name.
# We also hard code the hidden_file here to reduce duplication.
#
# We should consider moving the "out" default pattern into here, many other
# consumers of the tf_gen_op_wrapper_py rule would be simplified if we don't
# hard code the ops/ directory.

def tf_gen_op_wrapper_private_py(
        name,
        out = None,
        deps = [],
        require_shape_functions = False,
        visibility = []):
    if not name.endswith("_gen"):
        fail("name must end in _gen")
    if not visibility:
        visibility = ["//visibility:private"]
    bare_op_name = name[:-4]  # Strip off the _gen
    tf_gen_op_wrapper_py(
        name = bare_op_name,
        out = out,
        visibility = visibility,
        deps = deps,
        require_shape_functions = require_shape_functions,
        generated_target_name = name,
        api_def_srcs = [
            "//tensorflow/core/api_def:base_api_def",
            "//tensorflow/core/api_def:python_api_def",
        ],
    )
