"""Module extension for rocm."""

load("@local_xla//third_party/gpus:rocm_configure.bzl", "rocm_configure")

rocm_configure_ext = module_extension(
    implementation = lambda mctx: rocm_configure(name = "local_config_rocm"),
)
