"""Package groups for XLA CPU NanoRt Users."""

def xla_cpu_nanort_packages(name = "xla_cpu_nanort_packages"):
    native.package_group(
        name = "nanort_users",
        packages = ["//..."],
    )
