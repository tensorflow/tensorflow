"""Repository rule to create a platform for a docker image to be used with RBE."""

def _remote_platform_configure_impl(repository_ctx):
    platform = repository_ctx.attr.platform
    if platform == "local":
        os = repository_ctx.os.name.lower()
        if os.startswith("windows"):
            platform = "windows"
        elif os.startswith("mac os"):
            platform = "osx"
        else:
            platform = "linux"

    cpu = "x86_64"
    machine_type = repository_ctx.execute(["bash", "-c", "echo $MACHTYPE"]).stdout
    if (machine_type.startswith("ppc") or
        machine_type.startswith("powerpc")):
        cpu = "ppc"
    elif machine_type.startswith("s390x"):
        cpu = "s390x"
    elif machine_type.startswith("aarch64"):
        cpu = "aarch64"
    elif machine_type.startswith("arm64"):
        cpu = "aarch64"
    elif machine_type.startswith("arm"):
        cpu = "arm"
    elif machine_type.startswith("mips64"):
        cpu = "mips64"
    elif machine_type.startswith("riscv64"):
        cpu = "riscv64"

    exec_properties = repository_ctx.attr.platform_exec_properties

    serialized_exec_properties = "{"
    for k, v in exec_properties.items():
        serialized_exec_properties += "\"%s\" : \"%s\"," % (k, v)
    serialized_exec_properties += "}"

    repository_ctx.template(
        "BUILD",
        Label("@local_tsl//third_party/remote_config:BUILD.tpl"),
        {
            "%{platform}": platform,
            "%{exec_properties}": serialized_exec_properties,
            "%{cpu}": cpu,
        },
    )

remote_platform_configure = repository_rule(
    implementation = _remote_platform_configure_impl,
    attrs = {
        "platform_exec_properties": attr.string_dict(mandatory = True),
        "platform": attr.string(default = "linux", values = ["linux", "windows", "local"]),
    },
)
