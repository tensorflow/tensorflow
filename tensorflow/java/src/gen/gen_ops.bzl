load(
    "//tensorflow:tensorflow.bzl",
    "tf_binary_additional_srcs",
)

# Generate Java wrapper classes for all registered core operations and package
# them into a single source archive (.srcjar).
#
# For example:
#  tf_java_op_gen_srcjar("gen_sources", ":gen_tool", "my.package")
#
# will create a genrule named "gen_sources" that generates source files under
#     ops/src/main/java/my/package/**/*.java
#
# and then archive those source files into
#     ops/gen_sources.srcjar
#
def tf_java_op_gen_srcjar(
        name,
        gen_tool,
        base_package,
        api_def_srcs = [],
        out_dir = "ops/",
        out_src_dir = "src/main/java/",
        visibility = ["//tensorflow/java:__pkg__"]):
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
                ") | cut -d\" \" -f1))",
            )
        api_def_args_str = ",".join(api_def_args)

    gen_cmds += ["$(location " + gen_tool + ")" +
                 " --output_dir=$(@D)/" + out_src_dir +
                 " --base_package=" + base_package +
                 " --api_dirs=" + api_def_args_str]

    # Generate a source archive containing generated code for these ops.
    gen_srcjar = out_dir + name + ".srcjar"
    gen_cmds += ["$(JAVABASE)/bin/jar cMf $(location :" + gen_srcjar + ") -C $(@D) src"]

    native.genrule(
        name = name,
        srcs = srcs,
        outs = [gen_srcjar],
        tools = [
            gen_tool,
        ] + tf_binary_additional_srcs(),
        toolchains = ["@bazel_tools//tools/jdk:current_host_java_runtime"],
        cmd = " && ".join(gen_cmds),
    )
