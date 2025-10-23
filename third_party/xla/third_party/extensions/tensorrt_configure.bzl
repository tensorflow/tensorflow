"""Module extension for tensorrt."""

load("@local_xla//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")

tensorrt_configure_ext = module_extension(
    implementation = lambda mctx: tensorrt_configure(name = "local_config_tensorrt"),
)
