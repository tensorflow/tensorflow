"""IFRT package_group definitions."""

def ifrt_package_groups(name = "ifrt_package_groups"):
    """Defines visibility groups for IFRT."""

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
