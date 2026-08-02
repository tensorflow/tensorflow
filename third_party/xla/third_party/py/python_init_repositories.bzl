"""
Legacy-compatible wrapper for Hermetic Python initialization.
See the WORKSPACE file for instructions on using the updated
Hermetic Python initialization.
"""

load(
    "@rules_ml_toolchain//py:python_init_repositories.bzl",
    _python_init_repositories = "python_init_repositories",
)

def python_init_repositories(
        requirements = {},
        local_wheel_workspaces = [],
        local_wheel_dist_folder = None,
        default_python_version = None,
        local_wheel_inclusion_list = ["*"],
        local_wheel_exclusion_list = []):
    _python_init_repositories(
        requirements = requirements,
        local_wheel_workspaces = local_wheel_workspaces,
        local_wheel_dist_folder = local_wheel_dist_folder,
        default_python_version = default_python_version,
        local_wheel_inclusion_list = local_wheel_inclusion_list,
        local_wheel_exclusion_list = local_wheel_exclusion_list,
    )
