"""A simple portable implementation of build_test."""

def build_test(name, targets, visibility = None):
    """Generates a test that just verifies that the specified targets can be built."""

    # Generate an sh_test rule that lists the specified targets as data,
    # (thus forcing those targets to be built before the test can be run)
    # and that runs a script which always succeeds.
    native.sh_test(
        name = name,
        srcs = [name + ".sh"],
        data = targets,
        visibility = visibility,
    )

    # Generate the script which always succeeds.  We just generate an empty script.
    native.genrule(
        name = name + "_gen_sh",
        outs = [name + ".sh"],
        cmd = "> $@",
        visibility = ["//visibility:private"],
    )
