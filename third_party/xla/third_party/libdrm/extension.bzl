"""Module extension for libdrm."""

load("//third_party/libdrm:workspace.bzl", libdrm = "repo")

def _libdrm_ext_impl(mctx):  # @unused
    libdrm()

libdrm_ext = module_extension(
    implementation = _libdrm_ext_impl,
)
