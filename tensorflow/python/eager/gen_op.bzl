"""For eager-mode Python."""

load("//tensorflow:tensorflow.bzl",
     "clean_dep",
     "tf_binary_additional_srcs",
     "tf_copts",
     "tf_cc_binary")

def tfe_gen_op_wrapper_py(name,
                          out=None,
                          visibility=None,
                          deps=[],
                          generated_target_name=None):
  """Generate an eager-mode Python op wrapper for an op library."""
  # Construct a cc_binary containing the specified ops.
  tool_name = "gen_" + name + "_py_wrappers_cc"
  if not deps:
    deps = [str(Label("//tensorflow/core:" + name + "_op_lib"))]
  tf_cc_binary(
      name=tool_name,
      linkopts=["-lm"],
      copts=tf_copts(),
      linkstatic=1,
      deps=([
          clean_dep("//tensorflow/python/eager:python_eager_op_gen_main")
      ] + deps),
      visibility=[clean_dep("//visibility:public")],)

  # Invoke the previous cc_binary to generate a python file.
  if not out:
    out = "gen_" + name + ".py"

  native.genrule(
      name=name + "_pygenrule",
      outs=[out],
      tools=[tool_name] + tf_binary_additional_srcs(),
      cmd=("$(location " + tool_name + ")  > $@"))

  # Make a py_library out of the generated python file.
  if not generated_target_name:
    generated_target_name = name
  native.py_library(
      name=generated_target_name,
      srcs=[out],
      srcs_version="PY2AND3",
      visibility=visibility,
      deps=[
          clean_dep("//tensorflow/python/eager:framework_for_generated_wrappers"),
      ],)
