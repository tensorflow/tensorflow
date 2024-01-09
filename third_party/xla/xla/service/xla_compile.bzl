"""Build macro that compile a Mhlo or StableHlo file into a Aot Result


To use from your BUILD file, add the following line to load the macro:

load("//xla/service:xla_compile.bzl", "xla_aot_compile_cpu", "xla_aot_compile_gpu")

Then call the macro like this:

xla_aot_compile(
    name = "test_aot_result",
    module = ":test_module_file",
)

"""

xla_compile_tool = "//xla/service:xla_compile"

def xla_aot_compile_cpu(
        name,
        module):
    """Runs xla_compile to compile an MHLO or StableHLO module into an AotCompilationResult for CPU

    Args:
        name: The name of the build rule.
        module: The MHLO or StableHLO file to compile.
    """

    # Run xla_compile to generate the file containing an AotCompilationResult.
    native.genrule(
        name = ("gen_" + name),
        srcs = [module],
        outs = [name],
        cmd = ("$(location " + xla_compile_tool + ")" +
               " --module_file=$(location " + module + ")" +
               " --output_file=$(location " + name + ")" +
               " --platform=cpu"),
        tools = [xla_compile_tool],
    )

    return

def xla_aot_compile_gpu(
        name,
        module,
        gpu_target_config,
        autotune_results):
    """Runs xla_compile to compile an MHLO, StableHLO or HLO module into an AotCompilationResult for GPU

    Args:
        name: The name of the build rule.
        module: The MHLO or StableHLO file to compile.
        gpu_target_config: The serialized GpuTargetConfigProto
        autotune_results: AOT AutotuneResults
    """

    # Run xla_compile to generate the file containing an AotCompilationResult.
    native.genrule(
        name = ("gen_" + name),
        srcs = [module, gpu_target_config, autotune_results],
        outs = [name],
        cmd = (
            "$(location " + xla_compile_tool + ")" +
            " --module_file=$(location " + module + ")" +
            " --output_file=$(location " + name + ")" +
            " --platform=gpu" +
            " --gpu_target_config=$(location " + gpu_target_config + ")" +
            " --autotune_results=$(location " + autotune_results + ")"
        ),
        tools = [xla_compile_tool],
        # copybara:comment_begin(oss-only)
        target_compatible_with = select({
            "@local_config_cuda//:is_cuda_enabled": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        # copybara:comment_end
    )

    return

def xla_aot_compile_gpu_runtime_autotuning(
        name,
        module,
        gpu_target_config):
    """Runs xla_compile to compile an MHLO or StableHLO module into an AotCompilationResult for GPU

    Args:
        name: The name of the build rule.
        module: The MHLO or StableHLO file to compile.
        gpu_target_config: The serialized GpuTargetConfigProto
    """

    # Run xla_compile to generate the file containing an AotCompilationResult.
    native.genrule(
        name = ("gen_" + name),
        srcs = [module, gpu_target_config],
        outs = [name],
        cmd = (
            "$(location " + xla_compile_tool + ")" +
            " --module_file=$(location " + module + ")" +
            " --output_file=$(location " + name + ")" +
            " --platform=gpu" +
            " --gpu_target_config=$(location " + gpu_target_config + ")"
        ),
        tools = [xla_compile_tool],
        # copybara:comment_begin(oss-only)
        target_compatible_with = select({
            "@local_config_cuda//:is_cuda_enabled": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        # copybara:comment_end
    )
