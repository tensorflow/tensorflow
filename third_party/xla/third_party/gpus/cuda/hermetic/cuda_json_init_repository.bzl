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

"""Hermetic CUDA redistributions JSON repository initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_mirror_urls")
load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_versions.bzl",
    "CUDA_REDIST_JSON_DICT",
    "CUDNN_REDIST_JSON_DICT",
    "MIRRORED_TARS_CUDA_REDIST_JSON_DICT",
    "MIRRORED_TARS_CUDNN_REDIST_JSON_DICT",
)

def _get_env_var(ctx, name):
    return ctx.os.environ.get(name)

def _get_json_file_content(
        repository_ctx,
        url_to_sha256,
        mirrored_tars_url_to_sha256,
        json_file_name,
        mirrored_tars_json_file_name):
    use_cuda_tars = _get_env_var(
        repository_ctx,
        "USE_CUDA_TAR_ARCHIVE_FILES",
    )
    (url, sha256) = url_to_sha256
    if mirrored_tars_url_to_sha256:
        (mirrored_tar_url, mirrored_tar_sha256) = mirrored_tars_url_to_sha256
    else:
        mirrored_tar_url = None
        mirrored_tar_sha256 = None
    json_file = None

    if use_cuda_tars and mirrored_tar_url:
        json_tar_downloaded = repository_ctx.download(
            url = mirrored_tar_url,
            sha256 = mirrored_tar_sha256,
            output = mirrored_tars_json_file_name,
            allow_fail = True,
        )
        if json_tar_downloaded.success:
            print("Successfully downloaded mirrored tar file: {}".format(
                mirrored_tar_url,
            ))  # buildifier: disable=print
            json_file = mirrored_tars_json_file_name
        else:
            print("Failed to download mirrored tar file: {}".format(
                mirrored_tar_url,
            ))  # buildifier: disable=print

    if not json_file:
        repository_ctx.download(
            url = tf_mirror_urls(url),
            sha256 = sha256,
            output = json_file_name,
        )
        json_file = json_file_name

    json_content = repository_ctx.read(repository_ctx.path(json_file))
    repository_ctx.delete(json_file)
    return json_content

def _cuda_redist_json_impl(repository_ctx):
    cuda_version = (_get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    _get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    local_cuda_path = _get_env_var(repository_ctx, "LOCAL_CUDA_PATH")
    cudnn_version = (_get_env_var(repository_ctx, "HERMETIC_CUDNN_VERSION") or
                     _get_env_var(repository_ctx, "TF_CUDNN_VERSION"))
    local_cudnn_path = _get_env_var(repository_ctx, "LOCAL_CUDNN_PATH")
    supported_cuda_versions = repository_ctx.attr.cuda_json_dict.keys()
    if (cuda_version and not local_cuda_path and
        (cuda_version not in supported_cuda_versions)):
        fail(
            ("The supported CUDA versions are {supported_versions}." +
             " Please provide a supported version in HERMETIC_CUDA_VERSION" +
             " environment variable or add JSON URL for" +
             " CUDA version={version}.")
                .format(
                supported_versions = supported_cuda_versions,
                version = cuda_version,
            ),
        )
    supported_cudnn_versions = repository_ctx.attr.cudnn_json_dict.keys()
    if cudnn_version and not local_cudnn_path and (cudnn_version not in supported_cudnn_versions):
        fail(
            ("The supported CUDNN versions are {supported_versions}." +
             " Please provide a supported version in HERMETIC_CUDNN_VERSION" +
             " environment variable or add JSON URL for" +
             " CUDNN version={version}.")
                .format(
                supported_versions = supported_cudnn_versions,
                version = cudnn_version,
            ),
        )
    cuda_redistributions = "{}"
    cudnn_redistributions = "{}"
    if cuda_version and not local_cuda_path:
        if cuda_version in repository_ctx.attr.mirrored_tars_cuda_json_dict.keys():
            mirrored_tars_url_to_sha256 = repository_ctx.attr.mirrored_tars_cuda_json_dict[cuda_version]
        else:
            mirrored_tars_url_to_sha256 = {}
        cuda_redistributions = _get_json_file_content(
            repository_ctx,
            url_to_sha256 = repository_ctx.attr.cuda_json_dict[cuda_version],
            mirrored_tars_url_to_sha256 = mirrored_tars_url_to_sha256,
            json_file_name = "redistrib_cuda_%s.json" % cuda_version,
            mirrored_tars_json_file_name = "redistrib_cuda_%s_tar.json" % cuda_version,
        )
    if cudnn_version and not local_cudnn_path:
        if cudnn_version in repository_ctx.attr.mirrored_tars_cudnn_json_dict.keys():
            mirrored_tars_url_to_sha256 = repository_ctx.attr.mirrored_tars_cudnn_json_dict[cudnn_version]
        else:
            mirrored_tars_url_to_sha256 = {}
        cudnn_redistributions = _get_json_file_content(
            repository_ctx,
            mirrored_tars_url_to_sha256 = mirrored_tars_url_to_sha256,
            url_to_sha256 = repository_ctx.attr.cudnn_json_dict[cudnn_version],
            json_file_name = "redistrib_cudnn_%s.json" % cudnn_version,
            mirrored_tars_json_file_name = "redistrib_cudnn_%s_tar.json" % cudnn_version,
        )

    repository_ctx.file(
        "distributions.bzl",
        """CUDA_REDISTRIBUTIONS = {cuda_redistributions}

CUDNN_REDISTRIBUTIONS = {cudnn_redistributions}
""".format(
            cuda_redistributions = cuda_redistributions,
            cudnn_redistributions = cudnn_redistributions,
        ),
    )
    repository_ctx.file(
        "BUILD",
        "",
    )

cuda_redist_json = repository_rule(
    implementation = _cuda_redist_json_impl,
    attrs = {
        "cuda_json_dict": attr.string_list_dict(mandatory = True),
        "cudnn_json_dict": attr.string_list_dict(mandatory = True),
        "mirrored_tars_cuda_json_dict": attr.string_list_dict(mandatory = True),
        "mirrored_tars_cudnn_json_dict": attr.string_list_dict(mandatory = True),
    },
    environ = [
        "HERMETIC_CUDA_VERSION",
        "HERMETIC_CUDNN_VERSION",
        "TF_CUDA_VERSION",
        "TF_CUDNN_VERSION",
        "LOCAL_CUDA_PATH",
        "LOCAL_CUDNN_PATH",
        "USE_CUDA_TAR_ARCHIVE_FILES",
    ],
)

def cuda_json_init_repository(
        cuda_json_dict = CUDA_REDIST_JSON_DICT,
        cudnn_json_dict = CUDNN_REDIST_JSON_DICT,
        mirrored_tars_cuda_json_dict = MIRRORED_TARS_CUDA_REDIST_JSON_DICT,
        mirrored_tars_cudnn_json_dict = MIRRORED_TARS_CUDNN_REDIST_JSON_DICT):
    cuda_redist_json(
        name = "cuda_redist_json",
        cuda_json_dict = cuda_json_dict,
        cudnn_json_dict = cudnn_json_dict,
        mirrored_tars_cuda_json_dict = mirrored_tars_cuda_json_dict,
        mirrored_tars_cudnn_json_dict = mirrored_tars_cudnn_json_dict,
    )
