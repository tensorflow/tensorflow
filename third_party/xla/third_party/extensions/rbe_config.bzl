"""Module extension to initialize RBE configs."""

load("//tools/toolchains/remote_config:configs.bzl", "initialize_rbe_configs")

rbe_config_ext = module_extension(
    implementation = lambda mctx: initialize_rbe_configs(),  # Generates `@ml_build_config_platform`
)
