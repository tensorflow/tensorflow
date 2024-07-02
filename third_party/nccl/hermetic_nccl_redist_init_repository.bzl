# Copyright 2024 The TensorFlow Authors. All rights reserved.
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

"""Hermetic NCCL repositories initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_mirror_urls")
load(
    "//third_party/gpus/cuda:hermetic_cuda_redist_init_repositories.bzl",
    "OS_ARCH_DICT",
    "get_archive_name",
    "get_env_var",
)

def _cuda_wheel_impl(repository_ctx):
    cuda_version = (get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    saved_major_version = ""
    if cuda_version:
        # Download archive only when GPU config is used.
        arch = OS_ARCH_DICT[repository_ctx.os.arch]
        dict_key = "{cuda_version}-{arch}".format(
            cuda_version = cuda_version,
            arch = arch,
        )
        supported_versions = repository_ctx.attr.url_dict.keys()
        if dict_key not in supported_versions:
            fail(
                ("The supported NCCL versions are {supported_versions}." +
                 " Please provide a supported version in HERMETIC_CUDA_VERSION" +
                 " environment variable or add NCCL distribution for" +
                 " CUDA version={version}, OS={arch}.")
                    .format(
                    supported_versions = supported_versions,
                    version = cuda_version,
                    arch = arch,
                ),
            )
        sha256 = repository_ctx.attr.sha256_dict[dict_key]
        url = repository_ctx.attr.url_dict[dict_key]
        saved_major_version = repository_ctx.attr.version_dict[dict_key].split(".")[0]

        archive_name = get_archive_name(url)
        file_name = archive_name + ".zip"

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

        repository_ctx.template(
            "BUILD",
            repository_ctx.attr.build_template,
            {
                "%{version}": saved_major_version,
            },
        )
    else:
        # If no CUDA version is found, use the dummy build file if present.
        repository_ctx.file(
            "BUILD",
            repository_ctx.read(repository_ctx.attr.dummy_build_file),
        )
    repository_ctx.file("version.txt", saved_major_version)

_cuda_wheel = repository_rule(
    implementation = _cuda_wheel_impl,
    attrs = {
        "sha256_dict": attr.string_dict(mandatory = True),
        "url_dict": attr.string_dict(mandatory = True),
        "version_dict": attr.string_dict(mandatory = True),
        "build_template": attr.label(),
        "dummy_build_file": attr.label(),
        "strip_prefix": attr.string(),
    },
    environ = ["HERMETIC_CUDA_VERSION", "TF_CUDA_VERSION"],
)

def cuda_wheel(name, sha256_dict, url_dict, version_dict, **kwargs):
    _cuda_wheel(
        name = name,
        sha256_dict = sha256_dict,
        url_dict = url_dict,
        version_dict = version_dict,
        **kwargs
    )

def hermetic_nccl_redist_init_repository(cuda_nccl_wheels):
    nccl_artifacts_dict = {"sha256_dict": {}, "url_dict": {}, "version_dict": {}}
    for cuda_version, nccl_wheel_info in cuda_nccl_wheels.items():
        for arch in OS_ARCH_DICT.values():
            if arch in nccl_wheel_info.keys():
                cuda_version_to_arch_key = "%s-%s" % (cuda_version, arch)
                nccl_artifacts_dict["sha256_dict"][cuda_version_to_arch_key] = nccl_wheel_info[arch].get("sha256", "")
                nccl_artifacts_dict["url_dict"][cuda_version_to_arch_key] = nccl_wheel_info[arch]["url"]
                nccl_artifacts_dict["version_dict"][cuda_version_to_arch_key] = nccl_wheel_info[arch]["version"]

    cuda_wheel(
        name = "cuda_nccl",
        sha256_dict = nccl_artifacts_dict["sha256_dict"],
        url_dict = nccl_artifacts_dict["url_dict"],
        version_dict = nccl_artifacts_dict["version_dict"],
        build_template = Label("//third_party/nccl:cuda_nccl.BUILD.tpl"),
        dummy_build_file = Label("//third_party/nccl:cuda_nccl_dummy.BUILD"),
        strip_prefix = "nvidia/nccl",
    )
