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

"""Hermetic NVSHMEM redistributions JSON repository initialization. Consult the WORKSPACE on how to use it."""

load(
    "//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "get_json_file_content",
)
load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_versions.bzl",
    "MIRRORED_TARS_NVSHMEM_REDIST_JSON_DICT",
    "NVSHMEM_REDIST_JSON_DICT",
)

def _get_env_var(ctx, name):
    return ctx.os.environ.get(name)

def _nvshmem_redist_json_impl(repository_ctx):
    nvshmem_version = _get_env_var(repository_ctx, "HERMETIC_NVSHMEM_VERSION")
    local_nvshmem_path = _get_env_var(repository_ctx, "LOCAL_NVSHMEM_PATH")
    supported_nvshmem_versions = repository_ctx.attr.nvshmem_json_dict.keys()
    if (nvshmem_version and not local_nvshmem_path and
        (nvshmem_version not in supported_nvshmem_versions)):
        fail(
            ("The supported NVSHMEM versions are {supported_versions}." +
             " Please provide a supported version in HERMETIC_NVSHMEM_VERSION" +
             " environment variable or add JSON URL for" +
             " NVSHMEM version={version}.")
                .format(
                supported_versions = supported_nvshmem_versions,
                version = nvshmem_version,
            ),
        )
    nvshmem_redistributions = "{}"
    if nvshmem_version and not local_nvshmem_path:
        if nvshmem_version in repository_ctx.attr.mirrored_tars_nvshmem_json_dict.keys():
            mirrored_tars_url_to_sha256 = repository_ctx.attr.mirrored_tars_nvshmem_json_dict[nvshmem_version]
        else:
            mirrored_tars_url_to_sha256 = {}
        nvshmem_redistributions = get_json_file_content(
            repository_ctx,
            url_to_sha256 = repository_ctx.attr.nvshmem_json_dict[nvshmem_version],
            mirrored_tars_url_to_sha256 = mirrored_tars_url_to_sha256,
            json_file_name = "redistrib_nvshmem_%s.json" % nvshmem_version,
            mirrored_tars_json_file_name = "redistrib_nvshmem_%s_tar.json" % nvshmem_version,
            use_tar_file_env_var_name = "USE_NVSHMEM_TAR_ARCHIVE_FILES",
        )

    repository_ctx.file(
        "distributions.bzl",
        "NVSHMEM_REDISTRIBUTIONS = {nvshmem_redistributions}".format(
            nvshmem_redistributions = nvshmem_redistributions,
        ),
    )
    repository_ctx.file(
        "BUILD",
        "",
    )

nvshmem_redist_json = repository_rule(
    implementation = _nvshmem_redist_json_impl,
    attrs = {
        "nvshmem_json_dict": attr.string_list_dict(mandatory = True),
        "mirrored_tars_nvshmem_json_dict": attr.string_list_dict(mandatory = True),
    },
    environ = [
        "HERMETIC_NVSHMEM_VERSION",
        "LOCAL_NVSHMEM_PATH",
        "USE_NVSHMEM_TAR_ARCHIVE_FILES",
    ],
)

def nvshmem_json_init_repository(
        nvshmem_json_dict = NVSHMEM_REDIST_JSON_DICT,
        mirrored_tars_nvshmem_json_dict = MIRRORED_TARS_NVSHMEM_REDIST_JSON_DICT):
    nvshmem_redist_json(
        name = "nvshmem_redist_json",
        nvshmem_json_dict = nvshmem_json_dict,
        mirrored_tars_nvshmem_json_dict = mirrored_tars_nvshmem_json_dict,
    )
