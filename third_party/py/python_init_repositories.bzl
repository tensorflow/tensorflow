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
