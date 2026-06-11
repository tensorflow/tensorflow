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
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
        target_triple = target_llvm_triple(),
        compatible_with = []):
    """Runs xla_compile to compile an MHLO or StableHLO module into an AotCompilationResult for CPU

    Args:
        name: The name of the build rule.
        module: The MHLO or StableHLO file to compile.
        target_cpu: The cpu name to compile for.
        target_triple: The target triple to compile for.
        target_features: The target features to compile for.
        compatible_with: Compatibility of the target.
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
        compatible_with = compatible_with,
    )

    return

def _xla_aot_cpu_validation(name, pb_target, deps, testonly, tags, compatible_with):
    """Dependency check genrule."""
    infer_tool = "//xla/backends/cpu/lite_aot:infer_lite_aot_deps_main"
    deps_str = " ".join(deps)
    validation_target = name + "_validation"

    native.genrule(
        name = validation_target,
        srcs = [":" + pb_target],
        outs = [name + ".validation_ok"],
        cmd = """
        $(location {infer_tool}) --compilation_result=$(location :{pb_target}) --output_deps=inferred_deps.txt
        MISSING_DEPS=""
        while read -r dep; do
            FOUND=0
            for actual_dep in {deps_str}; do
                if [ "$$dep" == "$$actual_dep" ]; then
                    FOUND=1
                    break
                fi
            done
            if [ $$FOUND -eq 0 ]; then
                MISSING_DEPS="$$MISSING_DEPS $$dep"
            fi
        done < inferred_deps.txt
        if [ -n "$$MISSING_DEPS" ]; then
            echo -e "\\033[1;31mERROR:\\033[0m XLA:CPU dependencies are missing from '{name}':"
            for dep in $$MISSING_DEPS; do
                echo "  $$dep"
            done
            exit 1
        fi
        touch $@
        """.format(
            infer_tool = infer_tool,
            pb_target = pb_target,
            deps_str = deps_str,
            name = name,
        ),
        tools = [infer_tool],
        testonly = testonly,
        tags = tags,
        compatible_with = compatible_with,
    )
    return validation_target

def _xla_aot_cpu_header(name, pb_target, model_factory_name, testonly, tags, compatible_with):
    """Header generation genrule."""
    header_target = name + ".h"
    gen_header_tool = "//xla/backends/cpu/lite_aot:generate_aot_header"

    # Compute the runtime path.
    pb_runtime_path = native.package_name() + "/" + pb_target if native.package_name() else pb_target

    native.genrule(
        name = name + "_header",
        srcs = [":" + pb_target],
        outs = [header_target],
        cmd = "$(location {gen_header_tool}) --pb_path={pb_path} --output_h=$@ --model_factory_name={model_factory_name}".format(
            gen_header_tool = gen_header_tool,
            pb_path = pb_runtime_path,
            model_factory_name = model_factory_name,
        ),
        tools = [gen_header_tool],
        testonly = testonly,
        tags = tags,
        compatible_with = compatible_with,
    )
    return header_target

def xla_aot_cpu_cc_library(
        name,
        module,
        deps = [],
        model_factory_name = None,
        visibility = None,
        testonly = None,
        tags = [],
        compatible_with = []):
    """Compiles an MHLO or StableHLO module into a cc_library for XLA:CPU Lite AOT.

    Args:
        name: The name of the cc_library target.
        module: The MHLO, HLO or StableHLO file to compile.
        deps: Additional dependencies for the cc_library.
        model_factory_name: The full name of the generated C++ function. Defaults to 'Get' + name +
        'AotFunction'.
        visibility: Visibility of the target.
        testonly: Whether the target is testonly.
        tags: Tags for the target.
        compatible_with: Compatibility of the target.
    """
    if not model_factory_name:
        model_factory_name = "Get" + name + "AotFunction"

    pb_target = name + "_aot"
    xla_aot_compile_cpu(
        name = pb_target,
        module = module,
        compatible_with = compatible_with,
    )

    validation_target = _xla_aot_cpu_validation(
        name = name,
        pb_target = pb_target,
        deps = deps,
        testonly = testonly,
        tags = tags,
        compatible_with = compatible_with,
    )

    header_target = _xla_aot_cpu_header(
        name = name,
        pb_target = pb_target,
        model_factory_name = model_factory_name,
        testonly = testonly,
        tags = tags,
        compatible_with = compatible_with,
    )

    cc_library(
        name = name,
        hdrs = [":" + header_target],
        deps = [
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/status:statusor",
            "@com_google_absl//absl/strings",
            "//xla/backends/cpu/lite_aot:xla_aot_function",
            "//xla/service/cpu:executable_proto_cc",
            "@tsl//tsl/platform:env",
            "@tsl//tsl/platform:path",
            "@tsl//tsl/platform:status",
        ] + deps,
        data = [
            ":" + pb_target,
            ":" + validation_target,  # We link the validation target to ensure the validation is run.
        ],
        visibility = visibility,
        testonly = testonly,
        tags = tags,
        compatible_with = compatible_with,
    )

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

def xla_aot_compile_gpu_for_platform(
        name,
        module,
        target_platform_version):
    """Runs xla_compile to compile an MHLO or StableHLO module into an AotCompilationResult for GPU.

    Args:
        name: The name of the build rule.
        module: The MHLO or StableHLO file to compile.
        target_platform_version: The name of the target platform version to compile for.
    """

    # Run xla_compile to generate the file containing an AotCompilationResult.
    native.genrule(
        name = ("gen_" + name),
        srcs = [module],
        outs = [name],
        cmd = (
            "$(location " + xla_compile_tool + ")" +
            " --module_file=$(location " + module + ")" +
            " --output_file=$(location " + name + ")" +
            " --platform=gpu" +
            " --target_platform_version=" + target_platform_version
        ),
        tools = [xla_compile_tool],
        # copybara:comment_begin(oss-only)
        target_compatible_with = select({
            "@local_config_cuda//:is_cuda_enabled": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        # copybara:comment_end
    )
