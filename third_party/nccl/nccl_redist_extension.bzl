"""Module extension for patched nccl_redist_init_repository."""

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

def _nccl_redist_ext_impl(mctx):
    nccl_redist_init_repository(
        patches = ["@org_tensorflow//third_party/nccl:nccl_wheel.patch"],
    )

nccl_redist_ext = module_extension(
    implementation = _nccl_redist_ext_impl,
)
