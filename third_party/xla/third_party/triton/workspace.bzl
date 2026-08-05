"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_mirror_urls")
load("//third_party/triton:common/series.bzl", "common_patch_list")
load(
    "//third_party/triton:intel_xpu/workspace.bzl",
    "XPU_TRITON_COMMIT",
    "XPU_TRITON_SHA256",
    "use_xpu_triton",
)
load("//third_party/triton:oss_only/series.bzl", "oss_only_patch_list")

# This is a custom repository rule (in place of the standard tf_http_archive)
# so that the Intel XPU backend source can be used in place of the upstream
# Triton when building XLA for the oneAPI backend (ENABLE_INTEL_XPU_TRITON=1).
def _triton_archive_impl(repository_ctx):
    patch_files = repository_ctx.attr.patch_file
    sha256 = repository_ctx.attr.sha256
    strip_prefix = repository_ctx.attr.strip_prefix
    urls = repository_ctx.attr.urls

    if use_xpu_triton(repository_ctx):
        sha256 = XPU_TRITON_SHA256
        patch_files = oss_only_patch_list + [
            "//third_party/triton:intel_xpu/intel_build.patch",
        ]
        strip_prefix = "intel-xpu-backend-for-triton-" + XPU_TRITON_COMMIT
        urls = tf_mirror_urls("https://github.com/intel/intel-xpu-backend-for-triton/archive/{}.tar.gz".format(XPU_TRITON_COMMIT))

    # Resolve labels before download_and_extract to prevent
    # unnecessary re-downloads. Borrowed from tf_http_archive.
    for patch_file in patch_files:
        repository_ctx.path(Label(patch_file))

    repository_ctx.download_and_extract(
        url = urls,
        sha256 = sha256,
        stripPrefix = strip_prefix,
    )
    for patch_file in patch_files:
        repository_ctx.patch(repository_ctx.path(Label(patch_file)), strip = 1)

triton_archive = repository_rule(
    implementation = _triton_archive_impl,
    attrs = {
        "patch_file": attr.string_list(),
        "sha256": attr.string(mandatory = True),
        "strip_prefix": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
    },
    environ = ["ENABLE_INTEL_XPU_TRITON"],
)

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "72259b1cc3c543c361dcd185a6ff89662e8ed52f"
    TRITON_SHA256 = "35744577b837c66cf934b3b1d31b1496e3c205c0fb431b8bdcc76f4c0245312c"
    triton_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-" + TRITON_COMMIT,
        urls = tf_mirror_urls("https://github.com/triton-lang/triton/archive/{}.tar.gz".format(TRITON_COMMIT)),
        patch_file = common_patch_list + oss_only_patch_list,
    )
