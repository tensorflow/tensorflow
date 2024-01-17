'''
 Repository rule to set some environment variables.
 Can be set via build parameter "--repo_env=<VARIABLE_NAME>=<value>"
 e.g "--repo_env=REQUIREMENTS_FILE_NAME=requirements.in"

 List of variables:
 REQUIREMENTS_FILE_NAME
'''

def _updater_config_repository_impl(repository_ctx):
    repository_ctx.file("BUILD", "")
    requirements_file_name = repository_ctx.os.environ.get("REQUIREMENTS_FILE_NAME", "requirements.in")
    repository_ctx.file(
        "updater_config_repository.bzl",
        "REQUIREMENTS_FILE_NAME = \"%s\"" %
        requirements_file_name,
    )

updater_config_repository = repository_rule(
    implementation = _updater_config_repository_impl,
    environ = ["REQUIREMENTS_FILE_NAME"],
)
