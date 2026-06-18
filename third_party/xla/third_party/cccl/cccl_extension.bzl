"""CCCL extension for Bazel modules."""

load("@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl", "cuda_redist_init_repositories")
load("//third_party/cccl:workspace.bzl", "CCCL_3_2_0_DIST_DICT", "CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES")

def _cccl_extension_impl(ctx):  # @unused
    cuda_redist_init_repositories(
        cuda_redistributions = CCCL_3_2_0_DIST_DICT,
        redist_versions_to_build_templates = CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES,
    )

cccl_extension = module_extension(
    implementation = _cccl_extension_impl,
)
