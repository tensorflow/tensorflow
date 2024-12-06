""" Repository rule to generate a file with python wheel version. """

def _python_wheel_version_repository_impl(repository_ctx):
    file_content = repository_ctx.read(
        repository_ctx.path(repository_ctx.attr.file_with_version),
    )
    version_line_start_index = file_content.find(repository_ctx.attr.version_key)
    version_line_end_index = version_line_start_index + file_content[version_line_start_index:].find("\n")
    repository_ctx.file(
        "wheel_version.bzl",
        file_content[version_line_start_index:version_line_end_index].replace(
            repository_ctx.attr.version_key,
            "WHEEL_VERSION",
        ),
    )
    repository_ctx.file("BUILD", "")

python_wheel_version_repository = repository_rule(
    implementation = _python_wheel_version_repository_impl,
    attrs = {
        "file_with_version": attr.label(mandatory = True, allow_single_file = True),
        "version_key": attr.string(mandatory = True),
    },
)
