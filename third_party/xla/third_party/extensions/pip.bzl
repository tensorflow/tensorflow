"""Module extension to install pip deps."""

load("@pypi//:requirements.bzl", "install_deps")

pypi_ext = module_extension(
    implementation = lambda mctx: install_deps(),
)
