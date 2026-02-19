"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("@rules_python//python:repositories.bzl", "py_repositories")
load("//third_party/py:python_repo.bzl", "python_repository")

def python_init_repositories(
        requirements = {},
        local_wheel_workspaces = [],
        local_wheel_dist_folder = None,
        default_python_version = None,
        local_wheel_inclusion_list = ["*"],
        local_wheel_exclusion_list = []):
    python_repository(
        name = "python_version_repo",
        requirements_versions = requirements.keys(),
        requirements_locks = requirements.values(),
        local_wheel_workspaces = local_wheel_workspaces,
        local_wheel_dist_folder = local_wheel_dist_folder,
        default_python_version = default_python_version,
        local_wheel_inclusion_list = local_wheel_inclusion_list,
        local_wheel_exclusion_list = local_wheel_exclusion_list,
    )
    py_repositories()

_configure_tag = tag_class(
    attrs = {
        "requirements_versions": attr.string_list(
            mandatory = False,
            default = [],
        ),
        "requirements_locks": attr.label_list(
            mandatory = False,
            default = [],
        ),
        "local_wheel_workspaces": attr.label_list(
            mandatory = False,
            default = [],
        ),
        "local_wheel_dist_folder": attr.string(
            mandatory = False,
            default = "",
        ),
        "default_python_version": attr.string(
            mandatory = False,
            default = "",
        ),
        "local_wheel_inclusion_list": attr.string_list(
            mandatory = False,
            default = ["*"],
        ),
        "local_wheel_exclusion_list": attr.string_list(
            mandatory = False,
            default = [],
        ),
    },
)

def _python_init_repositories_ext_impl(mctx):
    kwargs = {}

    for mod in mctx.modules:
        for tag in mod.tags.configure:
            if tag.requirements_versions:
                kwargs["requirements_versions"] = tag.requirements_versions
            if tag.requirements_locks:
                kwargs["requirements_locks"] = tag.requirements_locks
            if tag.local_wheel_workspaces:
                kwargs["local_wheel_workspaces"] = tag.local_wheel_workspaces
            if tag.local_wheel_dist_folder:
                kwargs["local_wheel_dist_folder"] = tag.local_wheel_dist_folder
            if tag.default_python_version:
                kwargs["default_python_version"] = tag.default_python_version
            if tag.local_wheel_inclusion_list:
                kwargs["local_wheel_inclusion_list"] = tag.local_wheel_inclusion_list
            if tag.local_wheel_exclusion_list:
                kwargs["local_wheel_exclusion_list"] = tag.local_wheel_exclusion_list

    python_repository(
        name = "python_version_repo",
        **kwargs
    )
    py_repositories()

python_init_repositories_ext = module_extension(
    implementation = _python_init_repositories_ext_impl,
    tag_classes = {"configure": _configure_tag},
)
