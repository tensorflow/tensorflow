"""Repository rule to create a platform for a docker image to be used with RBE."""

def _remote_platform_configure_impl(repository_ctx):
    repository_ctx.template(
        "BUILD",
        Label("@org_tensorflow//third_party/remote_config:BUILD.tpl"),
        {
            "%{container_image}": repository_ctx.attr.container_image,
        },
    )

remote_platform_configure = repository_rule(
    implementation = _remote_platform_configure_impl,
    attrs = {
        "container_image": attr.string(mandatory = True),
    },
)
