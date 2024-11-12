""" Macros to unpack a wheel and use its content as a py_library. """

def _unpacked_wheel_impl(ctx):
    output_dir = ctx.actions.declare_directory(ctx.label.name)
    libs = []
    for dep in ctx.attr.deps:
        linker_inputs = dep[CcInfo].linking_context.linker_inputs.to_list()
        for linker_input in linker_inputs:
            if linker_input.libraries and linker_input.libraries[0].dynamic_library:
                lib = linker_input.libraries[0].dynamic_library
                libs.append(lib)
    wheel = None
    for w in ctx.files.wheel_rule_outputs:
        if w.basename.endswith(".whl"):
            wheel = w
            break
    script = """
    {zipper} x {wheel} -d {output}
    for lib in {libs}; do
        cp $lib {output}/tensorflow
    done
    """.format(
        zipper = ctx.executable.zipper.path,
        wheel = wheel.path,
        output = output_dir.path,
        libs = " ".join(["'%s'" % lib.path for lib in libs]),
    )
    ctx.actions.run_shell(
        inputs = ctx.files.wheel_rule_outputs + libs,
        command = script,
        outputs = [output_dir],
        tools = [ctx.executable.zipper],
    )

    return [
        DefaultInfo(files = depset([output_dir])),
    ]

_unpacked_wheel = rule(
    implementation = _unpacked_wheel_impl,
    attrs = {
        "wheel_rule_outputs": attr.label(mandatory = True, allow_files = True),
        "zipper": attr.label(
            default = Label("@bazel_tools//tools/zip:zipper"),
            cfg = "exec",
            executable = True,
        ),
        "deps": attr.label_list(providers = [CcInfo]),
    },
)

def wheel_library(name, wheel, deps = [], wheel_deps = []):
    unpacked_wheel_name = name + "_unpacked_wheel"
    _unpacked_wheel(
        name = unpacked_wheel_name,
        wheel_rule_outputs = wheel,
        deps = wheel_deps,
    )
    native.py_library(
        name = name,
        data = [":" + unpacked_wheel_name],
        imports = [unpacked_wheel_name],
        deps = deps,
        visibility = ["//visibility:public"],
    )
