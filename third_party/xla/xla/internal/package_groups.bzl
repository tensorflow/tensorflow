"""Package groups for XLA internal."""

def xla_internal_packages(name = "xla_internal_packages"):
    # DO NOT EXPAND THIS LIST. These files are for internal use only. External users should use
    # the public PJRT API instead.
    native.package_group(
        name = "hwi_internal",
        packages = [
            "//xla/internal/...",
        ],
    )
