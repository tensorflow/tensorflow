"""IFRT IR package_group definitions."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def ifrt_ir_package_groups(name = "ifrt_ir_package_groups"):
    """Defines visibility groups for IFRT IR.

    Args:
      name: The name of the package group.
    """

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
