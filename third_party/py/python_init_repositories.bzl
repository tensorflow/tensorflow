"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("@rules_python//python:repositories.bzl", "py_repositories")
load("//third_party/py:python_repo.bzl", "python_repository")

def python_init_repositories(
        requirements = {},
        local_wheel_workspaces = [],
        local_wheel_dist_folder = None):
    python_repository(
        name = "python_version_repo",
        requirements_versions = requirements.keys(),
        requirements_locks = requirements.values(),
        local_wheel_workspaces = local_wheel_workspaces,
        local_wheel_dist_folder = local_wheel_dist_folder,
    )
    py_repositories()
