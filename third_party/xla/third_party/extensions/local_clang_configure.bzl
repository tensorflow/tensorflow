"""Module extension for local clang."""

load("//third_party/clang_toolchain:local_clang_configure.bzl", "local_clang_configure")

local_clang_configure_ext = module_extension(
    implementation = lambda mctx: local_clang_configure(name = "local_config_clang"),
)
