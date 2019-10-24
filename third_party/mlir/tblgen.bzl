"""BUILD extensions for MLIR table generation."""

def gentbl(name, tblgen, td_file, tbl_outs, td_srcs = [], td_includes = [], strip_include_prefix = None, test = False):
    """gentbl() generates tabular code from a table definition file.

    Args:
      name: The name of the build rule for use in dependencies.
      tblgen: The binary used to produce the output.
      td_file: The primary table definitions file.
      tbl_outs: A list of tuples (opts, out), where each opts is a string of
        options passed to tblgen, and the out is the corresponding output file
        produced.
      td_srcs: A list of table definition files included transitively.
      td_includes: A list of include paths for relative includes.
      strip_include_prefix: attribute to pass through to cc_library.
      test: whether to create a test to invoke the tool too.
    """
    srcs = []
    srcs += td_srcs
    if td_file not in td_srcs:
        srcs += [td_file]

    td_includes_cmd = ["-I external/local_config_mlir/include -I external/org_tensorflow"]
    for td_include in td_includes:
        td_includes_cmd += ["-I%s" % td_include]
    local_inc = "-I $$(dirname $(location %s))" % td_file

    if test:
        # Rule to generate shell script to invoke tblgen. This generates a very
        # bare shell file which the sh_test uses.
        native.genrule(
            name = "%s_genrule_sh" % name,
            srcs = srcs,
            outs = ["%s.gen.sh" % name],
            cmd = ("echo \"\\$$1\" %s \\$${@:2} -o /dev/null > $@" % local_inc),
            executable = 1,
        )

    for (opts, out) in tbl_outs:
        # All arguments to generate the output except output destination.
        base_args = [
            "$(location %s)" % tblgen,
            "%s" % opts,
            "$(location %s)" % td_file,
            "-I$(GENDIR)",
        ] + td_includes_cmd
        rule_suffix = "_".join(opts.replace("-", "_").replace("=", "_").split(" "))

        # Rule to generate code using generated shell script.
        native.genrule(
            name = "%s_%s_genrule" % (name, rule_suffix),
            srcs = srcs,
            outs = [out],
            tools = [tblgen],
            message = "Generating code from table: %s" % td_file,
            cmd = (" ".join(base_args) + " %s -o $@" % local_inc),
        )

        # Optionally generate rule to test tblgen invocation.
        if test:
            native.sh_test(
                name = "%s_%s_genrule_test" % (name, rule_suffix),
                srcs = ["%s.gen.sh" % name],
                args = base_args,
                data = srcs + [tblgen],
            )

    # List of opts that do not generate cc files.
    skip_opts = ["-gen-op-doc"]
    hdrs = [f for (opts, f) in tbl_outs if opts not in skip_opts]
    native.cc_library(
        name = name,
        # include_prefix does not apply to textual_hdrs.
        hdrs = hdrs if strip_include_prefix else [],
        strip_include_prefix = strip_include_prefix,
        textual_hdrs = hdrs,
    )
