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

"""Hermetic CUDA repositories initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_mirror_urls")

OS_ARCH_DICT = {
    "amd64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
}
_REDIST_ARCH_DICT = {
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-sbsa": "aarch64-unknown-linux-gnu",
}

SUPPORTED_ARCHIVE_EXTENSIONS = [".zip", ".jar", ".war", ".aar", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.zst", ".tzst", ".tar.bz2", ".tbz", ".ar", ".deb", ".whl"]

def get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

def _get_file_name(url):
    last_slash_index = url.rfind("/")
    return url[last_slash_index + 1:]

def get_archive_name(url):
    filename = _get_file_name(url)
    for extension in SUPPORTED_ARCHIVE_EXTENSIONS:
        if filename.endswith(extension):
            return filename[:-len(extension)]
    return filename

def _cuda_http_archive_impl(repository_ctx):
    cuda_or_cudnn_version = None
    dist_version = ""
    saved_major_version = ""
    saved_major_minor_version = ""
    cuda_version = (get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    cudnn_version = (get_env_var(repository_ctx, "HERMETIC_CUDNN_VERSION") or
                     get_env_var(repository_ctx, "TF_CUDNN_VERSION"))
    if repository_ctx.attr.is_cudnn_dist:
        cuda_or_cudnn_version = cudnn_version
    else:
        cuda_or_cudnn_version = cuda_version
    if cuda_or_cudnn_version:
        # Download archive only when GPU config is used.
        dist_version = repository_ctx.attr.dist_version
        arch_key = OS_ARCH_DICT[repository_ctx.os.arch]
        if arch_key not in repository_ctx.attr.url_dict.keys():
            arch_key = "cuda{version}_{arch}".format(
                version = cuda_version.split(".")[0],
                arch = arch_key,
            )
        if arch_key in repository_ctx.attr.url_dict.keys():
            (url, sha256) = repository_ctx.attr.url_dict[arch_key]

            # If url is not relative, then appending prefix is not needed.
            if not (url.startswith("http") or url.startswith("file:///")):
                if repository_ctx.attr.is_cudnn_dist:
                    url = repository_ctx.attr.cudnn_dist_path_prefix + url
                else:
                    url = repository_ctx.attr.cuda_dist_path_prefix + url
            archive_name = get_archive_name(url)
            file_name = _get_file_name(url)

            repository_ctx.download(
                url = tf_mirror_urls(url),
                output = file_name,
                sha256 = sha256,
            )
            if repository_ctx.attr.override_strip_prefix:
                strip_prefix = repository_ctx.attr.override_strip_prefix
            else:
                strip_prefix = archive_name
            repository_ctx.extract(
                archive = file_name,
                stripPrefix = strip_prefix,
            )
            repository_ctx.delete(file_name)

            if repository_ctx.attr.build_template:
                version_to_list = dist_version.split(".") if dist_version else ""
                if len(version_to_list) > 0:
                    saved_major_version = version_to_list[0]
                    saved_major_minor_version = (version_to_list[0] +
                                                 "." + version_to_list[1])
                build_template = repository_ctx.attr.build_template

                # Workaround for CUDA 11 distribution versions.
                if cuda_version and cuda_version.startswith("11"):
                    if saved_major_version == "11":
                        if repository_ctx.name == "cuda_cudart":
                            saved_major_version = "11.0"
                        if repository_ctx.name == "cuda_cupti":
                            saved_major_version = cuda_version

                repository_ctx.template(
                    "BUILD",
                    build_template,
                    {
                        "%{version}": saved_major_version,
                        "%{major_minor_version}": saved_major_minor_version,
                    },
                )
            else:
                repository_ctx.file(
                    "BUILD",
                    repository_ctx.read(repository_ctx.attr.build_file),
                )
        else:
            # If no matching arch is found, use the dummy build file.
            repository_ctx.file(
                "BUILD",
                repository_ctx.read(repository_ctx.attr.dummy_build_file),
            )
    else:
        # If no CUDA or CUDNN version is found, use the dummy build file if present.
        repository_ctx.file(
            "BUILD",
            repository_ctx.read(repository_ctx.attr.dummy_build_file or
                                repository_ctx.attr.build_file),
        )
    repository_ctx.file("version.txt", saved_major_version)

_cuda_http_archive = repository_rule(
    implementation = _cuda_http_archive_impl,
    attrs = {
        "dist_version": attr.string(mandatory = True),
        "url_dict": attr.string_list_dict(mandatory = True),
        "build_template": attr.label(),
        "dummy_build_file": attr.label(),
        "build_file": attr.label(),
        "is_cudnn_dist": attr.bool(),
        "override_strip_prefix": attr.string(),
        "cudnn_dist_path_prefix": attr.string(),
        "cuda_dist_path_prefix": attr.string(),
    },
    environ = [
        "HERMETIC_CUDA_VERSION",
        "HERMETIC_CUDNN_VERSION",
        "TF_CUDA_VERSION",
        "TF_CUDNN_VERSION",
    ],
)

def cuda_http_archive(name, dist_version, url_dict, **kwargs):
    _cuda_http_archive(
        name = name,
        dist_version = dist_version,
        url_dict = url_dict,
        **kwargs
    )

def _get_distribution_urls(dist_info):
    url_dict = {}
    for arch in _REDIST_ARCH_DICT.keys():
        if "relative_path" not in dist_info[arch]:
            if "full_path" not in dist_info[arch]:
                for cuda_version, data in dist_info[arch].items():
                    # CUDNN JSON might contain paths for each CUDA version.
                    path_key = "relative_path"
                    if path_key not in data.keys():
                        path_key = "full_path"
                    url_dict["{cuda_version}_{arch}" \
                        .format(
                        cuda_version = cuda_version,
                        arch = _REDIST_ARCH_DICT[arch],
                    )] = [data[path_key], data.get("sha256", "")]
            else:
                url_dict[_REDIST_ARCH_DICT[arch]] = [
                    dist_info[arch]["full_path"],
                    dist_info[arch].get("sha256", ""),
                ]
        else:
            url_dict[_REDIST_ARCH_DICT[arch]] = [
                dist_info[arch]["relative_path"],
                dist_info[arch].get("sha256", ""),
            ]
    return url_dict

def _get_cuda_archive(
        repo_name,
        dist_dict,
        dist_name,
        cuda_dist_path_prefix = "",
        cudnn_dist_path_prefix = "",
        build_file = None,
        build_template = None,
        dummy_build_file = None,
        is_cudnn_dist = False):
    if dist_name in dist_dict.keys():
        return cuda_http_archive(
            name = repo_name,
            dist_version = dist_dict[dist_name]["version"],
            build_file = build_file,
            build_template = build_template,
            dummy_build_file = dummy_build_file,
            url_dict = _get_distribution_urls(dist_dict[dist_name]),
            is_cudnn_dist = is_cudnn_dist,
            cuda_dist_path_prefix = cuda_dist_path_prefix,
            cudnn_dist_path_prefix = cudnn_dist_path_prefix,
        )
    else:
        return cuda_http_archive(
            name = repo_name,
            dist_version = "",
            build_file = build_file,
            build_template = build_template,
            dummy_build_file = dummy_build_file,
            url_dict = {"": []},
            is_cudnn_dist = is_cudnn_dist,
        )

def hermetic_cudnn_redist_init_repository(cudnn_dist_path_prefix, cudnn_distributions):
    cudnn_build_template = Label("//third_party/gpus/cuda:cuda_cudnn.BUILD.tpl")
    cudnn_dummy_build_file = Label("//third_party/gpus/cuda:cuda_cudnn_dummy.BUILD")
    if "cudnn" in cudnn_distributions.keys():
        cudnn_version = cudnn_distributions["cudnn"]["version"]
        if cudnn_version.startswith("9"):
            cudnn_build_template = Label("//third_party/gpus/cuda:cuda_cudnn9.BUILD.tpl")
            cudnn_dummy_build_file = Label("//third_party/gpus/cuda:cuda_cudnn9_dummy.BUILD")
    _get_cuda_archive(
        repo_name = "cuda_cudnn",
        dist_dict = cudnn_distributions,
        dist_name = "cudnn",
        build_template = cudnn_build_template,
        dummy_build_file = cudnn_dummy_build_file,
        is_cudnn_dist = True,
        cudnn_dist_path_prefix = cudnn_dist_path_prefix,
    )

def hermetic_cuda_redist_init_repositories(
        cuda_distributions,
        cuda_dist_path_prefix):
    _get_cuda_archive(
        repo_name = "cuda_cccl",
        dist_dict = cuda_distributions,
        dist_name = "cuda_cccl",
        build_file = Label("//third_party/gpus/cuda:cuda_cccl.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cublas",
        dist_dict = cuda_distributions,
        dist_name = "libcublas",
        build_template = Label("//third_party/gpus/cuda:cuda_cublas.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cublas_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cudart",
        dist_dict = cuda_distributions,
        dist_name = "cuda_cudart",
        build_template = Label("//third_party/gpus/cuda:cuda_cudart.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cudart_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cufft",
        dist_dict = cuda_distributions,
        dist_name = "libcufft",
        build_template = Label("//third_party/gpus/cuda:cuda_cufft.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cufft_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cupti",
        dist_dict = cuda_distributions,
        dist_name = "cuda_cupti",
        build_template = Label("//third_party/gpus/cuda:cuda_cupti.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cupti_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_curand",
        dist_dict = cuda_distributions,
        dist_name = "libcurand",
        build_template = Label("//third_party/gpus/cuda:cuda_curand.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_curand_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cusolver",
        dist_dict = cuda_distributions,
        dist_name = "libcusolver",
        build_template = Label("//third_party/gpus/cuda:cuda_cusolver.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cusolver_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cusparse",
        dist_dict = cuda_distributions,
        dist_name = "libcusparse",
        build_template = Label("//third_party/gpus/cuda:cuda_cusparse.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cusparse_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvcc",
        dist_dict = cuda_distributions,
        dist_name = "cuda_nvcc",
        build_file = Label("//third_party/gpus/cuda:cuda_nvcc.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvjitlink",
        dist_dict = cuda_distributions,
        dist_name = "libnvjitlink",
        build_template = Label("//third_party/gpus/cuda:cuda_nvjitlink.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_nvjitlink_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvml",
        dist_dict = cuda_distributions,
        dist_name = "cuda_nvml_dev",
        build_file = Label("//third_party/gpus/cuda:cuda_nvml.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvprune",
        dist_dict = cuda_distributions,
        dist_name = "cuda_nvprune",
        build_file = Label("//third_party/gpus/cuda:cuda_nvprune.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvrtc",
        dist_dict = cuda_distributions,
        dist_name = "cuda_nvrtc",
        build_template = Label("//third_party/gpus/cuda:cuda_nvrtc.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_nvrtc_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvtx",
        dist_dict = cuda_distributions,
        dist_name = "cuda_nvtx",
        build_file = Label("//third_party/gpus/cuda:cuda_nvtx.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
