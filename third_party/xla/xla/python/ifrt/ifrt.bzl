"""IFRT package_group definitions."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def ifrt_package_groups(name = "ifrt_package_groups"):
    """Defines visibility groups for IFRT.

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

    native.package_group(
        name = "serdes_any_version_users",
        packages = ["//..."],
    )

    native.package_group(
        name = "serdes_week_4_old_version_users",
        packages = ["//..."],
    )
