# Copyright 2025 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hermetic NVSHMEM repository initialization. Consult the WORKSPACE on how to use it."""

load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "OS_ARCH_DICT",
    "create_build_file",
    "create_dummy_build_file",
    "create_version_file",
    "download_redistribution",
    "get_env_var",
    "get_lib_name_to_version_dict",
    "get_major_library_version",
    "get_platform_architecture",
    "get_redistribution_urls",
    "get_version_and_template_lists",
    "use_local_path",
)
load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_versions.bzl",
    "MIRRORED_TAR_NVSHMEM_REDIST_PATH_PREFIX",
    "NVSHMEM_REDIST_PATH_PREFIX",
    "NVSHMEM_REDIST_VERSIONS_TO_BUILD_TEMPLATES",
)

def _use_local_nvshmem_path(repository_ctx, local_nvshmem_path):
    # buildifier: disable=function-docstring-args
    """ Creates symlinks and initializes hermetic NVSHMEM repository."""
    use_local_path(repository_ctx, local_nvshmem_path, ["include", "lib", "bin"])

def _use_downloaded_nvshmem_redistribution(repository_ctx):
    # buildifier: disable=function-docstring-args
    """ Downloads NVSHMEM redistribution and initializes hermetic NVSHMEM repository."""
    nvshmem_version = None
    major_version = ""
    nvshmem_version = get_env_var(repository_ctx, "HERMETIC_NVSHMEM_VERSION")
    cuda_version = (get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    if not nvshmem_version:
        # If no NVSHMEM version is found, comment out cc_import targets.
        create_dummy_build_file(repository_ctx)
        create_version_file(repository_ctx, major_version)
        return

    if len(repository_ctx.attr.url_dict) == 0:
        print("{} is not found in redistributions list.".format(
            repository_ctx.name,
        ))  # buildifier: disable=print
        create_dummy_build_file(repository_ctx)
        create_version_file(repository_ctx, major_version)
        return

    # Download archive only when GPU config is used.
    arch_key = OS_ARCH_DICT[get_platform_architecture(repository_ctx)]
    if arch_key not in repository_ctx.attr.url_dict.keys():
        arch_key = "cuda{version}_{arch}".format(
            version = cuda_version.split(".")[0],
            arch = arch_key,
        )
    if arch_key not in repository_ctx.attr.url_dict.keys():
        fail(
            ("The supported platforms are {supported_platforms}." +
             " Platform {platform} is not supported for {dist_name}.")
                .format(
                supported_platforms = repository_ctx.attr.url_dict.keys(),
                platform = arch_key,
                dist_name = repository_ctx.name,
            ),
        )

    download_redistribution(
        repository_ctx,
        arch_key,
        repository_ctx.attr.nvshmem_redist_path_prefix,
        repository_ctx.attr.mirrored_tar_nvshmem_redist_path_prefix,
        use_tar_file_env_var_name = "USE_NVSHMEM_TAR_ARCHIVE_FILES",
    )

    lib_name_to_version_dict = get_lib_name_to_version_dict(repository_ctx)
    major_version = get_major_library_version(
        repository_ctx,
        lib_name_to_version_dict,
    )
    create_build_file(
        repository_ctx,
        lib_name_to_version_dict,
        major_version,
    )

    create_version_file(repository_ctx, major_version)

def _nvshmem_repo_impl(repository_ctx):
    local_nvshmem_path = get_env_var(repository_ctx, "LOCAL_NVSHMEM_PATH")
    if local_nvshmem_path:
        _use_local_nvshmem_path(repository_ctx, local_nvshmem_path)
    else:
        _use_downloaded_nvshmem_redistribution(repository_ctx)

nvshmem_repo = repository_rule(
    implementation = _nvshmem_repo_impl,
    attrs = {
        "url_dict": attr.string_list_dict(mandatory = True),
        "versions": attr.string_list(mandatory = True),
        "build_templates": attr.label_list(mandatory = True),
        "override_strip_prefix": attr.string(),
        "nvshmem_redist_path_prefix": attr.string(),
        "mirrored_tar_nvshmem_redist_path_prefix": attr.string(),
    },
    environ = [
        "HERMETIC_CUDA_VERSION",
        "TF_CUDA_VERSION",
        "HERMETIC_NVSHMEM_VERSION",
        "LOCAL_NVSHMEM_PATH",
        "USE_NVSHMEM_TAR_ARCHIVE_FILES",
        "NVSHMEM_REDIST_TARGET_PLATFORM",
    ],
)

def nvshmem_redist_init_repository(
        nvshmem_redistributions,
        nvshmem_redist_path_prefix = NVSHMEM_REDIST_PATH_PREFIX,
        mirrored_tar_nvshmem_redist_path_prefix = MIRRORED_TAR_NVSHMEM_REDIST_PATH_PREFIX,
        redist_versions_to_build_templates = NVSHMEM_REDIST_VERSIONS_TO_BUILD_TEMPLATES):
    # buildifier: disable=function-docstring-args
    """Initializes NVSHMEM repository."""
    if "libnvshmem" in nvshmem_redistributions.keys():
        url_dict = get_redistribution_urls(nvshmem_redistributions["libnvshmem"])
    else:
        url_dict = {}
    repo_data = redist_versions_to_build_templates["libnvshmem"]
    versions, templates = get_version_and_template_lists(
        repo_data["version_to_template"],
    )
    nvshmem_repo(
        name = repo_data["repo_name"],
        versions = versions,
        build_templates = templates,
        url_dict = url_dict,
        nvshmem_redist_path_prefix = nvshmem_redist_path_prefix,
        mirrored_tar_nvshmem_redist_path_prefix = mirrored_tar_nvshmem_redist_path_prefix,
    )
