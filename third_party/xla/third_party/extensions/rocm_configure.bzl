"""Module extension for rocm."""

load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")

def _rocm_configure_ext_impl(_mctx):
    rocm_configure(
        name = "local_config_rocm",
        rocm_dist = "@config_rocm_hipcc//rocm:rocm_redist",
    )

rocm_configure_ext = module_extension(
    implementation = _rocm_configure_ext_impl,
)
