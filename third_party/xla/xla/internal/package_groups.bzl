"""Package groups for XLA internal."""

def xla_internal_packages(name = "xla_internal_packages"):
    native.package_group(
        name = "hwi_internal",
        packages = ["//..."],
    )
