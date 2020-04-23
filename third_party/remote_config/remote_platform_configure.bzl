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
    exec_properties = repository_ctx.attr.platform_exec_properties

    serialized_exec_properties = "{"
    for k, v in exec_properties.items():
        serialized_exec_properties += "\"%s\" : \"%s\"," % (k, v)
    serialized_exec_properties += "}"

    repository_ctx.template(
        "BUILD",
        Label("@org_tensorflow//third_party/remote_config:BUILD.tpl"),
        {
            "%{platform}": platform,
            "%{exec_properties}": serialized_exec_properties,
        },
    )

remote_platform_configure = repository_rule(
    implementation = _remote_platform_configure_impl,
    attrs = {
        "platform_exec_properties": attr.string_dict(mandatory = True),
        "platform": attr.string(default = "linux", values = ["linux", "windows", "local"]),
    },
)
