""" Macros to unpack a wheel and use its content as a py_library. """

def _unpacked_wheel_impl(ctx):
    output_dir = ctx.actions.declare_directory(ctx.label.name)
    ctx.actions.run(
        inputs = [ctx.file.wheel],
        outputs = [output_dir],
        executable = ctx.executable.zipper,
        arguments = ["x", ctx.file.wheel.path, "-d", output_dir.path],
    )
    return [
        DefaultInfo(files = depset([output_dir])),
    ]

_unpacked_wheel = rule(
    implementation = _unpacked_wheel_impl,
    attrs = {
        "wheel": attr.label(mandatory = True, allow_single_file = True),
        "zipper": attr.label(
            default = Label("@bazel_tools//tools/zip:zipper"),
            cfg = "exec",
            executable = True,
        ),
    },
)

def wheel_library(name, wheel, deps):
    unpacked_wheel_name = name + "_unpacked_wheel"
    _unpacked_wheel(
        name = unpacked_wheel_name,
        wheel = wheel,
    )
    native.py_library(
        name = name,
        data = [":" + unpacked_wheel_name],
        imports = [unpacked_wheel_name],
        deps = deps,
        visibility = ["//visibility:public"],
    )
