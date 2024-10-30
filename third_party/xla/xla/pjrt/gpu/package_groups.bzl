"""Package groups for XLA GPU internal."""

def xla_gpu_internal_packages(name = "xla_gpu_internal_packages"):
    native.package_group(
        name = "legacy_gpu_client_users",
        packages = ["//..."],
    )

    native.package_group(
        name = "legacy_gpu_topology_users",
        packages = ["//..."],
    )

    native.package_group(
        name = "legacy_gpu_internal_users",
        packages = ["//..."],
    )

    native.package_group(
        name = "legacy_se_gpu_pjrt_compiler_users",
        packages = ["//..."],
    )
