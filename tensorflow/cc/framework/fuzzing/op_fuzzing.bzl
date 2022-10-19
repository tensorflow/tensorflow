"""Functions for automatically generating fuzzers."""

load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows",
    "lrt_if_needed",
    "tf_cc_binary",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_test",
)

def tf_gen_op_wrapper_fuzz(
        name,
        out_ops_file,
        pkg = "",
        deps = None,
        include_internal_ops = 0,
        api_def_srcs = []):
    """
    Generates a file with fuzzers for a subset of ops.

    Args:
        name: name of the op class
        out_ops_file: prefix for file generation
        pkg: where to find op registrations
        deps: depedencies
        include_internal_ops: true if we should generate internal ops
        api_def_srcs: which op definitions to use
    """
    tool = out_ops_file + "_gen_fuzz"

    if deps == None:
        deps = [pkg + ":" + name + "_op_lib"]
    tf_cc_binary(
        name = tool,
        copts = tf_copts(),
        linkopts = if_not_windows(["-lm", "-Wl,-ldl"]) + lrt_if_needed(),
        linkstatic = 1,  # Faster to link this one-time-use binary dynamically
        deps = [
            "//tensorflow/cc/framework/fuzzing:cc_op_fuzz_gen_main",
        ] + deps,
    )

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
                " $$(dirname $$(echo $(locations " + api_def_src +
                ") | cut -d\" \" -f1))",
            )
        api_def_args_str = ",".join(api_def_args)

    out_fuzz_file = out_ops_file + "_fuzz.cc"
    native.genrule(
        name = name + "_genrule",
        outs = [
            out_fuzz_file,
        ],
        srcs = srcs,
        tools = [":" + tool],  # + tf_binary_additional_srcs(),
        cmd = ("$(location :" + tool + ") $(location :" + out_fuzz_file + ") " +
               str(include_internal_ops) + " " + api_def_args_str),
    )

def tf_gen_op_wrappers_fuzz(
        name,
        op_lib_names = [],
        pkg = "",
        deps = [
            "//tensorflow/cc:ops",
            "//tensorflow/cc:scope",
            "//tensorflow/cc:const_op",
        ],
        include_internal_ops = 0,
        api_def_srcs = [],
        extra_gen_deps = []):
    """
    Generates fuzzers for several groups of ops.

    Args:
        name: the name of the fuzz artifact
        op_lib_names: which op libraries to fuzz
        pkg: where to find op registrations
        deps: dependencies
        include_internal_ops: true if we should generate internal ops
        api_def_srcs: where to find the op definitions
        extra_gen_deps: extra dependencies for generation
    """
    fuzzsrcs = []
    for n in op_lib_names:
        tf_gen_op_wrapper_fuzz(
            n,
            "fuzzers/" + n,
            api_def_srcs = api_def_srcs,
            include_internal_ops = include_internal_ops,
            pkg = pkg,
            deps = [pkg + ":" + n + "_op_lib"] + extra_gen_deps,
        )
        fuzzsrcs.append("fuzzers/" + n + "_fuzz.cc")
    cc_test(
        name = name,
        srcs = fuzzsrcs,
        deps = deps +
               [
                   "//tensorflow/security/fuzzing/cc:fuzz_session",
                   "@com_google_googletest//:gtest_main",
                   "@com_google_fuzztest//fuzztest",
               ],
    )
