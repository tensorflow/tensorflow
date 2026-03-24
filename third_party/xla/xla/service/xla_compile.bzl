"""Build macro that compile a Mhlo or StableHlo file into a Aot Result


To use from your BUILD file, add the following line to load the macro:

load("//xla/service:xla_compile.bzl", "xla_aot_compile_cpu", "xla_aot_compile_gpu")

Then call the macro like this:

xla_aot_compile(
    name = "test_aot_result",
    module = ":test_module_file",
)

"""

load("//xla:xla.default.bzl", "xla_compile_target_cpu")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

xla_compile_tool = "//xla/service:xla_compile"

def target_llvm_triple():
    """Returns the target LLVM triple to be used for compiling the target."""

    # For triples of other targets, see:
    # http://llvm.org/docs/doxygen/html/Triple_8h_source.html
    return select({
        "//xla/tsl:android_arm": "armv7-none-android",
        "//xla/tsl:ios": "arm64-none-ios",
        "//xla/tsl:ios_x86_64": "x86_64-apple-ios",
        "//xla/tsl:linux_ppc64le": "ppc64le-ibm-linux-gnu",
        "//xla/tsl:linux_aarch64": "aarch64-none-linux-gnu",
        "//xla/tsl:macos_x86_64": "x86_64-none-darwin",
        "//xla/tsl:macos_arm64": "aarch64-none-darwin",
        "//xla/tsl:windows": "x86_64-none-windows",
        "//xla/tsl:linux_s390x": "systemz-none-linux-gnu",
        "//xla/tsl:arm_any": "aarch64-none-linux-gnu",
        "//conditions:default": "x86_64-pc-linux",
    })

def xla_aot_compile_cpu(
        name,
        module,
        target_cpu = xla_compile_target_cpu(),
        target_features = "",
        target_triple = target_llvm_triple()):
    """Runs xla_compile to compile an MHLO or StableHLO module into an AotCompilationResult for CPU

    Args:
        name: The name of the build rule.
        module: The MHLO or StableHLO file to compile.
        target_cpu: The cpu name to compile for.
        target_triple: The target triple to compile for.
        target_features: The target features to compile for.
    """

    # Run xla_compile to generate the file containing an AotCompilationResult.
    srcs = [module]

    cmd = ("$(location " + xla_compile_tool + ")" +
           " --module_file=$(location " + module + ")" +
           " --output_file=$(location " + name + ")" +
           " --platform=cpu" +
           " --target_cpu=" + target_cpu +
           " --target_triple=" + target_triple +
           " --target_features=" + target_features)

    native.genrule(
        name = ("gen_" + name),
        srcs = srcs,
        outs = [name],
        cmd = cmd,
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
