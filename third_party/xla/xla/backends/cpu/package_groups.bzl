"""Package groups for XLA:CPU backend internal access."""

# Integrations should use PJRT as the API to access XLA.
def xla_cpu_backend_access(name = "xla_cpu_backend_access"):
    native.package_group(
        name = "xla_backend_cpu_internal_access",
        packages = ["//..."],
    )
