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

load("//third_party:repo.bzl", "tf_mirror_urls")
load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "OS_ARCH_DICT",
    "create_build_file",
    "create_dummy_build_file",
    "create_version_file",
    "get_archive_name",
    "get_env_var",
    "get_lib_name_to_version_dict",
    "get_major_library_version",
    "get_version_and_template_lists",
    "use_local_path",
)
load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_versions.bzl",
    "CUDA_NVSHMEM_WHEELS",
    "REDIST_VERSIONS_TO_BUILD_TEMPLATES",
)

def _use_downloaded_nvshmem_wheel(repository_ctx):
    # buildifier: disable=function-docstring-args
    """ Downloads NVSHMEM wheel and inits hermetic NVSHMEM repository."""
    cuda_version = (get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    major_version = ""
    if not cuda_version:
        # If no CUDA version is found, comment out cc_import targets.
        create_dummy_build_file(repository_ctx)
        create_version_file(repository_ctx, major_version)
        return

    # Download archive only when GPU config is used.
    target_arch = get_env_var(repository_ctx, "CUDA_REDIST_TARGET_PLATFORM")
    if target_arch:
        if target_arch in OS_ARCH_DICT.keys():
            arch = OS_ARCH_DICT[target_arch]
        else:
            fail(
                "Unsupported architecture: {arch}, use one of {supported}".format(
                    arch = target_arch,
                    supported = OS_ARCH_DICT.keys(),
                ),
            )
    else:
        arch = OS_ARCH_DICT[repository_ctx.os.arch]
    dict_key = "{cuda_version}-{arch}".format(
        cuda_version = cuda_version,
        arch = arch,
    )
    supported_versions = repository_ctx.attr.url_dict.keys()
    if dict_key not in supported_versions:
        fail(
            ("The supported NVSHMEM versions are {supported_versions}." +
             " Please provide a supported version in HERMETIC_NVSHMEM_VERSION" +
             " environment variable or add NVSHMEM distribution for" +
             " CUDA version={version}, OS={arch}.")
                .format(
                supported_versions = supported_versions,
                version = cuda_version,
                arch = arch,
            ),
        )
    sha256 = repository_ctx.attr.sha256_dict[dict_key]
    url = repository_ctx.attr.url_dict[dict_key]

    archive_name = get_archive_name(url)
    file_name = archive_name + ".zip"

    print("Downloading and extracting {}".format(url))  # buildifier: disable=print
    repository_ctx.download(
        url = tf_mirror_urls(url),
        output = file_name,
        sha256 = sha256,
    )
    repository_ctx.extract(
        archive = file_name,
        stripPrefix = repository_ctx.attr.strip_prefix,
    )
    repository_ctx.delete(file_name)

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

def _use_local_nvshmem_path(repository_ctx, local_nvshmem_path):
    # buildifier: disable=function-docstring-args
    """ Creates symlinks and initializes hermetic NVSHMEM repository."""
    use_local_path(repository_ctx, local_nvshmem_path, ["include", "lib"])

def _nvidia_nvshmem_repo_impl(repository_ctx):
    local_nvshmem_path = get_env_var(repository_ctx, "LOCAL_NVSHMEM_PATH")
    if local_nvshmem_path:
        _use_local_nvshmem_path(repository_ctx, local_nvshmem_path)
    else:
        _use_downloaded_nvshmem_wheel(repository_ctx)

nvidia_nvshmem_repo = repository_rule(
    implementation = _nvidia_nvshmem_repo_impl,
    attrs = {
        "sha256_dict": attr.string_dict(mandatory = True),
        "url_dict": attr.string_dict(mandatory = True),
        "versions": attr.string_list(mandatory = True),
        "build_templates": attr.label_list(mandatory = True),
        "strip_prefix": attr.string(),
    },
    environ = [
        "HERMETIC_CUDA_VERSION",
        "TF_CUDA_VERSION",
        "LOCAL_NVSHMEM_PATH",
        "CUDA_REDIST_TARGET_PLATFORM",
    ],
)

def nvshmem_redist_init_repository(
        cuda_nvshmem_wheels = CUDA_NVSHMEM_WHEELS,
        redist_versions_to_build_templates = REDIST_VERSIONS_TO_BUILD_TEMPLATES):
    # buildifier: disable=function-docstring-args
    """Initializes NVSHMEM repository."""
    nvshmem_artifacts_dict = {"sha256_dict": {}, "url_dict": {}}
    for cuda_version, nvshmem_wheel_info in cuda_nvshmem_wheels.items():
        for arch in OS_ARCH_DICT.values():
            if arch in nvshmem_wheel_info.keys():
                cuda_version_to_arch_key = "%s-%s" % (cuda_version, arch)
                nvshmem_artifacts_dict["sha256_dict"][cuda_version_to_arch_key] = nvshmem_wheel_info[arch].get("sha256", "")
                nvshmem_artifacts_dict["url_dict"][cuda_version_to_arch_key] = nvshmem_wheel_info[arch]["url"]
    repo_data = redist_versions_to_build_templates["libnvshmem"]
    versions, templates = get_version_and_template_lists(
        repo_data["version_to_template"],
    )
    nvidia_nvshmem_repo(
        name = repo_data["repo_name"],
        sha256_dict = nvshmem_artifacts_dict["sha256_dict"],
        url_dict = nvshmem_artifacts_dict["url_dict"],
        versions = versions,
        build_templates = templates,
        strip_prefix = "nvidia/nvshmem",
    )
