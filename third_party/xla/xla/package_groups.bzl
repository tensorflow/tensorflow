"""XLA package_group definitions"""

def xla_package_groups(name = "xla_package_groups"):
    """Defines visibility groups for XLA.

    Args:
     name: package groups name
    """

    native.package_group(
        name = "friends",
        packages = ["//..."],
    )

    native.package_group(
        name = "internal",
        packages = ["//..."],
    )

    native.package_group(
        name = "backends",
        packages = ["//..."],
    )

    native.package_group(
        name = "codegen",
        packages = ["//..."],
    )

    native.package_group(
        name = "collectives",
        packages = ["//..."],
    )

    native.package_group(
        name = "runtime",
        packages = ["//..."],
    )

def xla_tests_package_groups(name = "xla_tests_package_groups"):
    """Defines visibility groups for XLA tests.

    Args:
     name: package groups name
    """

    native.package_group(
        name = "friends",
        packages = ["//..."],
    )
