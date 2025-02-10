"""PjRt-IFRT package_group definitions."""

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
