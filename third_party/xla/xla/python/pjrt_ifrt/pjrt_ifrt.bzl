"""PjRt-IFRT package_group definitions."""

load("//xla/tsl:tsl_core.bzl", "xla_bzl_visibility")

visibility(xla_bzl_visibility([
    "platforms/xla/...",
    "third_party/tensorflow/...",
]))

def pjrt_ifrt_package_groups(name = "pjrt_ifrt_package_groups"):
    """Defines visibility groups for PjRt-IFRT."""

    native.package_group(
        name = "users",
        packages = ["//..."],
    )

    native.package_group(
        name = "friends",
        packages = ["//..."],
    )

    native.package_group(
        name = "internal",
        packages = ["//..."],
    )
