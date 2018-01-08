# -*- Python -*-

load("//tensorflow:tensorflow.bzl",
     "tf_binary_additional_srcs",
     "tf_cc_binary",
     "tf_copts")

# Given a list of "ops_libs" (a list of files in the core/ops directory
# without their .cc extensions), generate Java wrapper code for all operations
# found in the ops files.
# Then, combine all those source files into a single archive (.srcjar).
#
# For example:
#  tf_java_op_gen_srcjar("gen_sources", "gen_tool", "my.package", [ "array_ops", "math_ops" ])
#
# will create a genrule named "gen_sources" that first generate source files:
#     ops/src/main/java/my/package/array/*.java
#     ops/src/main/java/my/package/math/*.java
#
# and then archive those source files in:
#     ops/gen_sources.srcjar
#
def tf_java_op_gen_srcjar(name,
                          gen_tool,
                          gen_base_package,
                          ops_libs=[],
                          ops_libs_pkg="//tensorflow/core",
                          out_dir="ops/",
                          out_src_dir="src/main/java/",
                          visibility=["//tensorflow/java:__pkg__"]):

  gen_tools = []
  gen_cmds = ["rm -rf $(@D)"]  # Always start from fresh when generating source files

  # Construct an op generator binary for each ops library.
  for ops_lib in ops_libs:
    gen_lib = ops_lib[:ops_lib.rfind("_")]
    out_gen_tool = out_dir + ops_lib + "_gen_tool"

    tf_cc_binary(
        name=out_gen_tool,
        copts=tf_copts(),
        linkopts=["-lm"],
        linkstatic=1,  # Faster to link this one-time-use binary dynamically
        deps=[gen_tool, ops_libs_pkg + ":" + ops_lib + "_op_lib"])

    gen_tools += [":" + out_gen_tool]
    gen_cmds += ["$(location :" + out_gen_tool + ")" +
                 " --output_dir=$(@D)/" + out_src_dir +
                 " --lib_name=" + gen_lib +
                 " --base_package=" + gen_base_package]

  # Generate a source archive containing generated code for these ops.
  gen_srcjar = out_dir + name + ".srcjar"
  gen_cmds += ["$(location @local_jdk//:jar) cMf $(location :" + gen_srcjar + ") -C $(@D) src"]
  gen_tools += ["@local_jdk//:jar"] + ["@local_jdk//:jdk"]
  gen_tools += tf_binary_additional_srcs()
  native.genrule(
      name=name,
      outs=[gen_srcjar],
      tools=gen_tools,
      cmd="&&".join(gen_cmds))
