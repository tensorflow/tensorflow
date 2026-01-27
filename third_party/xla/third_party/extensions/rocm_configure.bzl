"""Module extension for rocm."""

load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")

rocm_configure_ext = module_extension(
    implementation = lambda mctx: rocm_configure(name = "local_config_rocm"),
)
