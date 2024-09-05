"""XLA package_group definitions"""

def xla_package_groups(name = "xla_package_groups"):
    native.package_group(
        name = "friends",
        packages = ["//..."],
    )

    native.package_group(
        name = "internal",
        packages = ["//..."],
    )

    native.package_group(
        name = "runtime",
        packages = ["//..."],
    )

def xla_tests_package_groups(name = "xla_tests_package_groups"):
    native.package_group(
        name = "friends",
        packages = ["//..."],
    )
