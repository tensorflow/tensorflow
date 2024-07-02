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

def _get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

def _get_json_file_content(repository_ctx, url_to_sha256, json_file_name):
    if len(url_to_sha256) > 1:
        (url, sha256) = url_to_sha256
    else:
        url = url_to_sha256[0]
        sha256 = ""
    repository_ctx.download(
        url = tf_mirror_urls(url),
        sha256 = sha256,
        output = json_file_name,
    )
    return repository_ctx.read(repository_ctx.path(json_file_name))

def _cuda_redist_json_impl(repository_ctx):
    cuda_version = (_get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    _get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    cudnn_version = (_get_env_var(repository_ctx, "HERMETIC_CUDNN_VERSION") or
                     _get_env_var(repository_ctx, "TF_CUDNN_VERSION"))
    supported_cuda_versions = repository_ctx.attr.cuda_json_dict.keys()
    if cuda_version and (cuda_version not in supported_cuda_versions):
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
    if cudnn_version and (cudnn_version not in supported_cudnn_versions):
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
    cuda_distributions = "{}"
    cudnn_distributions = "{}"
    if cuda_version:
        cuda_distributions = _get_json_file_content(
            repository_ctx,
            repository_ctx.attr.cuda_json_dict[cuda_version],
            "redistrib_cuda_%s.json" % cuda_version,
        )
    if cudnn_version:
        cudnn_distributions = _get_json_file_content(
            repository_ctx,
            repository_ctx.attr.cudnn_json_dict[cudnn_version],
            "redistrib_cudnn_%s.json" % cudnn_version,
        )

    repository_ctx.file(
        "distributions.bzl",
        """CUDA_DISTRIBUTIONS = {cuda_distributions}

CUDNN_DISTRIBUTIONS = {cudnn_distributions}
""".format(
            cuda_distributions = cuda_distributions,
            cudnn_distributions = cudnn_distributions,
        ),
    )
    repository_ctx.file(
        "BUILD",
        "",
    )

_cuda_redist_json = repository_rule(
    implementation = _cuda_redist_json_impl,
    attrs = {
        "cuda_json_dict": attr.string_list_dict(mandatory = True),
        "cudnn_json_dict": attr.string_list_dict(mandatory = True),
    },
    environ = [
        "HERMETIC_CUDA_VERSION",
        "HERMETIC_CUDNN_VERSION",
        "TF_CUDA_VERSION",
        "TF_CUDNN_VERSION",
    ],
)

def cuda_redist_json(name, cuda_json_dict, cudnn_json_dict):
    _cuda_redist_json(
        name = name,
        cuda_json_dict = cuda_json_dict,
        cudnn_json_dict = cudnn_json_dict,
    )

def hermetic_cuda_json_init_repository(cuda_json_dict, cudnn_json_dict):
    cuda_redist_json(
        name = "cuda_redist_json",
        cuda_json_dict = cuda_json_dict,
        cudnn_json_dict = cudnn_json_dict,
    )
