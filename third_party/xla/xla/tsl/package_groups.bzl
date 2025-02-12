"""TSL package_group definitions"""

def tsl_package_groups(name = "tsl_package_groups"):
    native.package_group(
        name = "internal",
        packages = ["//..."],
    )
