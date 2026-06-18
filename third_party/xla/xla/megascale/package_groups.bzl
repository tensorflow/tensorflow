"""Megascale package_group definitions"""

def megascale_package_groups(name = "megascale_package_groups"):
    native.package_group(
        name = "internal",
        packages = ["//..."],
    )
