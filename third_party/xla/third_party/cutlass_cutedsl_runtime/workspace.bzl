"""Pinned CuTeDSL runtime artifact published with CUTLASS releases."""

load("//third_party:repo.bzl", "tf_mirror_urls")
load(
    "//third_party/gpus:nvidia_common_rules.bzl",
    "get_cuda_version",
    "get_env_var",
)

CUTLASS_CUTEDSL_RUNTIME_VERSION = "4.6.1"

# Non-CUDA builds can still analyze the test-only FFI target. CUDA 12 is the
# oldest artifact in this release and is used only when CUDA is not configured.
_DEFAULT_CUDA_MAJOR = "12"

_ARTIFACTS = {
    "aarch64-cuda12": "6d09d85ade64973d91cc70b05a2067ba00873874c6ecc3095c1309216ce904c2",
    "aarch64-cuda13": "8dfe0548abf69e96db2fc6fa5c9cc253de43122d8b10a79f476e0e46380a0c2e",
    "x86_64-cuda12": "f3480065a0ed5beb916577b44b2853511d0a9a6dad1846a5941324986947bc24",
    "x86_64-cuda13": "94b356f853ea409fb0aa857a962b18c0c756aee24947e430101bf4f2657c5247",
}

_HOST_ARCHITECTURES = {
    "aarch64": "aarch64",
    "amd64": "x86_64",
    "x86_64": "x86_64",
}

_TARGET_ARCHITECTURES = {
    "linux-aarch64": "aarch64",
    "linux-sbsa": "aarch64",
    "linux-x86_64": "x86_64",
}

def _architecture(repository_ctx):
    target = get_env_var(repository_ctx, "CUDA_REDIST_TARGET_PLATFORM")
    if target:
        if target not in _TARGET_ARCHITECTURES:
            fail("Unsupported CUDA_REDIST_TARGET_PLATFORM: {}".format(target))
        return _TARGET_ARCHITECTURES[target]

    host = repository_ctx.os.arch
    if host not in _HOST_ARCHITECTURES:
        fail("Unsupported host architecture: {}".format(host))
    return _HOST_ARCHITECTURES[host]

def _cutlass_cutedsl_runtime_impl(repository_ctx):
    build_file = repository_ctx.path(repository_ctx.attr.build_file)
    architecture = _architecture(repository_ctx)
    cuda_version = get_cuda_version(repository_ctx)
    cuda_major = (
        cuda_version.split(".")[0] if cuda_version else _DEFAULT_CUDA_MAJOR
    )
    artifact = "{}-cuda{}".format(architecture, cuda_major)
    if artifact not in _ARTIFACTS:
        fail(
            "Unsupported CuTeDSL runtime platform {}; supported platforms: {}".format(
                artifact,
                sorted(_ARTIFACTS.keys()),
            ),
        )

    archive = "cutlass-install-{}-cu{}-{}.tar.gz".format(
        architecture,
        cuda_major,
        CUTLASS_CUTEDSL_RUNTIME_VERSION,
    )
    repository_ctx.download_and_extract(
        url = tf_mirror_urls(
            "https://github.com/NVIDIA/cutlass/releases/download/v{}/{}".format(
                CUTLASS_CUTEDSL_RUNTIME_VERSION,
                archive,
            ),
        ),
        sha256 = _ARTIFACTS[artifact],
        stripPrefix = "{}/cu{}".format(architecture, cuda_major),
    )
    repository_ctx.symlink(build_file, "BUILD.bazel")

_cutlass_cutedsl_runtime = repository_rule(
    implementation = _cutlass_cutedsl_runtime_impl,
    attrs = {
        "build_file": attr.label(allow_single_file = True, mandatory = True),
    },
    environ = [
        "CUDA_REDIST_TARGET_PLATFORM",
        "HERMETIC_CUDA_VERSION",
        "TF_CUDA_VERSION",
    ],
)

def repo():
    _cutlass_cutedsl_runtime(
        name = "cutlass_cutedsl_runtime",
        build_file = "//third_party/cutlass_cutedsl_runtime:cutlass_cutedsl_runtime.BUILD",
    )
