"""Build macro that compile a Mhlo file into a Aot Result


To use from your BUILD file, add the following line to load the macro:

load("//tensorflow/compiler/xla/service:xla_compile.bzl", "xla_aot_compile")

Then call the macro like this:

xla_aot_compile(
    name = "test_aot_result",
    mhlo = ":test_mhlo_file",
)

"""

def xla_aot_compile(
        name,
        mhlo):
    """Runs xla_compile to compile a MHLO module into an AotCompilationResult for CPU

    Args:
        name: The name of the build rule.
        mhlo: The MHLO file to compile.
    """

    # Run xla_compile to generate the file containing an AotCompilationResult.
    xla_compile_tool = "//tensorflow/compiler/xla/service:xla_compile"

    native.genrule(
        name = ("gen_xla_compile"),
        srcs = [mhlo],
        outs = [name],
        cmd = ("$(location " + xla_compile_tool + ")" +
               " --mhlo_file=$(location " + mhlo + ")" +
               " --output_file=$(location " + name + ")"),
        tools = [xla_compile_tool],
    )

    return
