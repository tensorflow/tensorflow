# -*- Python -*-

load(
    "//tensorflow:tensorflow.bzl",
    "tf_binary_additional_srcs",
    "tf_cc_binary",
    "tf_copts",
)

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
                          api_def_srcs=[],
                          visibility=["//tensorflow/java:__pkg__"]):

  gen_cmds = ["rm -rf $(@D)"]  # Always start from fresh when generating source files
  srcs = api_def_srcs[:]

  if not api_def_srcs:
    api_def_args_str = ","
  else:
    api_def_args = []
    for api_def_src in api_def_srcs:
      # Add directory of the first ApiDef source to args.
      # We are assuming all ApiDefs in a single api_def_src are in the
      # same directory.
      api_def_args.append(
          "$$(dirname $$(echo $(locations " + api_def_src +
          ") | cut -d\" \" -f1))")
    api_def_args_str = ",".join(api_def_args)

  gen_tool_deps = [":java_op_gen_lib"]
  for ops_lib in ops_libs:
    gen_tool_deps.append(ops_libs_pkg + ":" + ops_lib + "_op_lib")

  tf_cc_binary(
      name=gen_tool,
      srcs=[
          "src/gen/cc/op_gen_main.cc",
      ],
      copts=tf_copts(),
      linkopts=["-lm"],
      linkstatic=1,  # Faster to link this one-time-use binary dynamically
      deps = gen_tool_deps)

  gen_cmds += ["$(location :" + gen_tool + ")" +
               " --output_dir=$(@D)/" + out_src_dir +
               " --base_package=" + gen_base_package +
               " --api_dirs=" + api_def_args_str]

  # Generate a source archive containing generated code for these ops.
  gen_srcjar = out_dir + name + ".srcjar"
  gen_cmds += ["$(location @local_jdk//:jar) cMf $(location :" + gen_srcjar + ") -C $(@D) src"]

  native.genrule(
      name=name,
      srcs=srcs,
      outs=[gen_srcjar],
      tools=[
          "@local_jdk//:jar",
          "@local_jdk//:jdk",
          gen_tool
      ] + tf_binary_additional_srcs(),
      cmd=" && ".join(gen_cmds))
