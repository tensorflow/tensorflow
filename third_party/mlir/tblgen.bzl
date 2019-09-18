"""BUILD extensions for MLIR table generation."""

def gentbl(name, tblgen, td_file, tbl_outs, td_srcs = [], td_includes = [], strip_include_prefix = None):
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
    """
    srcs = []
    srcs += td_srcs
    if td_file not in td_srcs:
        srcs += [td_file]

    # Add google_mlir/include directory as include so derived op td files can
    # import relative to that.
    td_includes_str = "-I external/local_config_mlir/include -I external/org_tensorflow "
    for td_include in td_includes:
        td_includes_str += "-I %s " % td_include
    td_includes_str += "-I $$(dirname $(location %s)) " % td_file
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
                td_includes_str,
                opts,
                td_file,
            )),
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
