""" Macros to unpack a wheel and use its content as a py_library. """

load("@rules_python//python:defs.bzl", "py_library")

def _unpacked_wheel_impl(ctx):
    output_dir = ctx.actions.declare_directory(ctx.label.name)
    wheel = ctx.file.wheel
    script = """
    {zipper} x {wheel} -d {output}
    for wheel_dep in {wheel_deps}; do
        {zipper} x $wheel_dep -d {output}
    done
    """.format(
        zipper = ctx.executable.zipper.path,
        wheel = wheel.path,
        output = output_dir.path,
        wheel_deps = " ".join([
            "'%s'" % wheel_dep.path
            for wheel_dep in ctx.files.wheel_deps
        ]),
    )
    ctx.actions.run_shell(
        inputs = ctx.files.wheel + ctx.files.wheel_deps,
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
        "wheel": attr.label(mandatory = True, allow_single_file = True),
        "zipper": attr.label(
            default = Label("@bazel_tools//tools/zip:zipper"),
            cfg = "exec",
            executable = True,
        ),
        "wheel_deps": attr.label_list(allow_files = True),
    },
)

def py_import(
        name,
        wheel,
        deps = [],
        wheel_deps = []):
    unpacked_wheel_name = name + "_unpacked_wheel"
    _unpacked_wheel(
        name = unpacked_wheel_name,
        wheel = wheel,
        wheel_deps = wheel_deps,
    )
    py_library(
        name = name,
        data = [":" + unpacked_wheel_name],
        imports = [unpacked_wheel_name],
        deps = deps,
        visibility = ["//visibility:public"],
    )

"""Unpacks the wheel and uses its content as a py_library.
Args:
  wheel: wheel file to unpack.
  deps: dependencies of the py_library.
  wheel_deps: additional wheels to unpack. These wheels will be unpacked in the
              same folder as the wheel.
"""  # buildifier: disable=no-effect
