"""Build macro that compile a Mhlo file into a Aot Result


To use from your BUILD file, add the following line to load the macro:

load("//tensorflow/compiler/xla/service:xla_compile.bzl", "xla_aot_compile_cpu", "xla_aot_compile_gpu")

Then call the macro like this:

xla_aot_compile(
    name = "test_aot_result",
    mhlo = ":test_mhlo_file",
)

"""

xla_compile_tool = "//tensorflow/compiler/xla/service:xla_compile"

def xla_aot_compile_cpu(
        name,
        mhlo):
    """Runs xla_compile to compile a MHLO module into an AotCompilationResult for CPU

    Args:
        name: The name of the build rule.
        mhlo: The MHLO file to compile.
    """

    # Run xla_compile to generate the file containing an AotCompilationResult.
    native.genrule(
        name = ("gen_" + name),
        srcs = [mhlo],
        outs = [name],
        cmd = ("$(location " + xla_compile_tool + ")" +
               " --mhlo_file=$(location " + mhlo + ")" +
               " --output_file=$(location " + name + ")" +
               " --platform=cpu"),
        tools = [xla_compile_tool],
    )

    return

def xla_aot_compile_gpu(
        name,
        mhlo,
        gpu_target_config):
    """Runs xla_compile to compile a MHLO module into an AotCompilationResult for GPU

    Args:
        name: The name of the build rule.
        mhlo: The MHLO file to compile.
        gpu_target_config: The serialized GpuTargetConfigProto
    """

    # Run xla_compile to generate the file containing an AotCompilationResult.
    native.genrule(
        name = ("gen_" + name),
        srcs = [mhlo, gpu_target_config],
        outs = [name],
        cmd = ("$(location " + xla_compile_tool + ")" +
               " --mhlo_file=$(location " + mhlo + ")" +
               " --output_file=$(location " + name + ")" +
               " --platform=gpu" +
               " --gpu_target_config=$(location " + gpu_target_config + ")"),
        tools = [xla_compile_tool],
    )

    return
