"""BUILD extensions for MLIR table generation."""

def gentbl(name, tblgen, td_file, tbl_outs, td_srcs = []):
    """gentbl() generates tabular code from a table definition file.

    Args:
      name: The name of the build rule for use in dependencies.
      tblgen: The binary used to produce the output.
      td_file: The primary table definitions file.
      tbl_outs: A list of tuples (opts, out), where each opts is a string of
        options passed to tblgen, and the out is the corresponding output file
        produced.
      td_srcs: A list of table definition files included transitively.
    """
    srcs = []
    srcs += td_srcs
    if td_file not in td_srcs:
        srcs += [td_file]

    # Add google_mlir/include directory as include so derived op td files can
    # import relative to that.
    td_includes = "-I external/local_config_mlir/include -I external/org_tensorflow "
    td_includes += "-I $$(dirname $(location %s)) " % td_file
    for (opts, out) in tbl_outs:
        rule_suffix = "_".join(opts.replace("-", "_").replace("=", "_").split(" "))
        native.genrule(
            name = "%s_%s_genrule" % (name, rule_suffix),
            srcs = srcs,
            outs = [out],
            tools = [tblgen],
            message = "Generating code from table: %s" % td_file,
            cmd = (("$(location %s) %s %s $(location %s) -o $@") % (
                tblgen,
                td_includes,
                opts,
                td_file,
            )),
        )

    native.cc_library(
        name = name,
        textual_hdrs = [f for (_, f) in tbl_outs],
    )
