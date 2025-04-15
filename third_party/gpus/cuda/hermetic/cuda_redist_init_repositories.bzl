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
load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_versions.bzl",
    "CUDA_REDIST_PATH_PREFIX",
    "CUDNN_REDIST_PATH_PREFIX",
    "REDIST_VERSIONS_TO_BUILD_TEMPLATES",
)

OS_ARCH_DICT = {
    "amd64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
    "tegra-aarch64": "tegra-aarch64-unknown-linux-gnu",
}
_REDIST_ARCH_DICT = {
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-sbsa": "aarch64-unknown-linux-gnu",
    "linux-aarch64": "tegra-aarch64-unknown-linux-gnu",
}

TEGRA = "tegra"

SUPPORTED_ARCHIVE_EXTENSIONS = [
    ".zip",
    ".jar",
    ".war",
    ".aar",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.xz",
    ".txz",
    ".tar.zst",
    ".tzst",
    ".tar.bz2",
    ".tbz",
    ".ar",
    ".deb",
    ".whl",
]

def get_env_var(ctx, name):
    return ctx.os.environ.get(name)

def _get_file_name(url):
    last_slash_index = url.rfind("/")
    return url[last_slash_index + 1:]

def get_archive_name(url):
    # buildifier: disable=function-docstring-return
    # buildifier: disable=function-docstring-args
    """Returns the archive name without extension."""
    filename = _get_file_name(url)
    for extension in SUPPORTED_ARCHIVE_EXTENSIONS:
        if filename.endswith(extension):
            return filename[:-len(extension)]
    return filename

LIB_EXTENSION = ".so."

def _get_lib_name_and_version(path):
    extension_index = path.rfind(LIB_EXTENSION)
    last_slash_index = path.rfind("/")
    lib_name = path[last_slash_index + 1:extension_index]
    lib_version = path[extension_index + len(LIB_EXTENSION):]
    return (lib_name, lib_version)

def _get_main_lib_name(repository_ctx):
    if repository_ctx.name == "cuda_driver":
        return "libcuda"
    else:
        return "lib{}".format(
            repository_ctx.name.split("_")[1],
        ).lower()

def _get_libraries_by_redist_name_in_dir(repository_ctx):
    lib_dir_path = repository_ctx.path("lib")
    if not lib_dir_path.exists:
        return []
    main_lib_name = _get_main_lib_name(repository_ctx)
    lib_dir_content = lib_dir_path.readdir()
    return [
        str(f)
        for f in lib_dir_content
        if (LIB_EXTENSION in str(f) and
            main_lib_name in str(f).lower())
    ]

def get_lib_name_to_version_dict(repository_ctx):
    # buildifier: disable=function-docstring-return
    # buildifier: disable=function-docstring-args
    """Returns a dict of library names and major versions."""
    lib_name_to_version_dict = {}
    for path in _get_libraries_by_redist_name_in_dir(repository_ctx):
        lib_name, lib_version = _get_lib_name_and_version(path)
        major_version_key = "%%{%s_version}" % lib_name.lower()
        minor_version_key = "%%{%s_minor_version}" % lib_name.lower()

        # We need to find either major or major.minor version if there is no
        # file with major version. E.g. if we have the following files:
        # libcudart.so
        # libcudart.so.12
        # libcudart.so.12.3.2,
        # we will save save {"%{libcudart_version}": "12",
        # "%{libcudart_minor_version}": "12.3.2"}
        if len(lib_version.split(".")) == 1:
            lib_name_to_version_dict[major_version_key] = lib_version
        if len(lib_version.split(".")) == 2:
            lib_name_to_version_dict[minor_version_key] = lib_version
            if (major_version_key not in lib_name_to_version_dict or
                len(lib_name_to_version_dict[major_version_key].split(".")) > 2):
                lib_name_to_version_dict[major_version_key] = lib_version
        if len(lib_version.split(".")) >= 3:
            if major_version_key not in lib_name_to_version_dict:
                lib_name_to_version_dict[major_version_key] = lib_version
            if minor_version_key not in lib_name_to_version_dict:
                lib_name_to_version_dict[minor_version_key] = lib_version
    return lib_name_to_version_dict

def create_dummy_build_file(repository_ctx, use_comment_symbols = True):
    repository_ctx.template(
        "BUILD",
        repository_ctx.attr.build_templates[0],
        {
            "%{multiline_comment}": "'''" if use_comment_symbols else "",
            "%{comment}": "#" if use_comment_symbols else "",
        },
    )

def _get_build_template(repository_ctx, major_lib_version):
    template = None
    for i in range(0, len(repository_ctx.attr.versions)):
        for dist_version in repository_ctx.attr.versions[i].split(","):
            if dist_version == major_lib_version:
                template = repository_ctx.attr.build_templates[i]
                break
    if not template:
        fail("No build template found for {} version {}".format(
            repository_ctx.name,
            major_lib_version,
        ))
    return template

def get_major_library_version(repository_ctx, lib_name_to_version_dict):
    # buildifier: disable=function-docstring-return
    # buildifier: disable=function-docstring-args
    """Returns the major library version provided the versions dict."""
    main_lib_name = _get_main_lib_name(repository_ctx)
    key = "%%{%s_version}" % main_lib_name
    if key not in lib_name_to_version_dict:
        return ""
    return lib_name_to_version_dict[key]

def create_build_file(
        repository_ctx,
        lib_name_to_version_dict,
        major_lib_version):
    # buildifier: disable=function-docstring-args
    """Creates a BUILD file for the repository."""
    if len(major_lib_version) == 0:
        build_template_content = repository_ctx.read(
            repository_ctx.attr.build_templates[0],
        )
        if "_version}" not in build_template_content:
            create_dummy_build_file(repository_ctx, use_comment_symbols = False)
        else:
            create_dummy_build_file(repository_ctx)
        return
    build_template = _get_build_template(
        repository_ctx,
        major_lib_version.split(".")[0],
    )
    repository_ctx.template(
        "BUILD",
        build_template,
        lib_name_to_version_dict | {
            "%{multiline_comment}": "",
            "%{comment}": "",
        },
    )

def _create_symlinks(repository_ctx, local_path, dirs):
    for dir in dirs:
        dir_path = "{path}/{dir}".format(
            path = local_path,
            dir = dir,
        )
        if not repository_ctx.path(local_path).exists:
            fail("%s directory doesn't exist!" % dir_path)
        repository_ctx.symlink(dir_path, dir)

def _create_libcuda_symlinks(
        repository_ctx,
        lib_name_to_version_dict):
    if repository_ctx.name == "cuda_driver":
        key = "%{libcuda_version}"
        if key not in lib_name_to_version_dict:
            return
        nvidia_driver_path = "lib/libcuda.so.{}".format(
            lib_name_to_version_dict[key],
        )
        if not repository_ctx.path(nvidia_driver_path).exists:
            fail("%s doesn't exist!" % nvidia_driver_path)
        repository_ctx.symlink(nvidia_driver_path, "lib/libcuda.so.1")
        repository_ctx.symlink("lib/libcuda.so.1", "lib/libcuda.so")

def _create_cuda_header_symlinks(repository_ctx):
    if repository_ctx.name == "cuda_nvcc":
        repository_ctx.symlink("../cuda_cudart/include/cuda.h", "include/cuda.h")

def _create_cuda_version_file(repository_ctx, lib_name_to_version_dict):
    key = "%{libcudart_version}"
    major_cudart_version = lib_name_to_version_dict[key] if key in lib_name_to_version_dict else ""
    if repository_ctx.name == "cuda_cudart":
        repository_ctx.file(
            "cuda_version.bzl",
            "MAJOR_CUDA_VERSION = \"{}\"".format(major_cudart_version),
        )

def create_version_file(repository_ctx, major_lib_version):
    repository_ctx.file(
        "version.bzl",
        "VERSION = \"{}\"".format(major_lib_version),
    )

def use_local_path(repository_ctx, local_path, dirs):
    # buildifier: disable=function-docstring-args
    """Creates repository using local redistribution paths."""
    _create_symlinks(
        repository_ctx,
        local_path,
        dirs,
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
    _create_libcuda_symlinks(
        repository_ctx,
        lib_name_to_version_dict,
    )
    _create_cuda_version_file(repository_ctx, lib_name_to_version_dict)
    create_version_file(repository_ctx, major_version)

def _use_local_cuda_path(repository_ctx, local_cuda_path):
    # buildifier: disable=function-docstring-args
    """ Creates symlinks and initializes hermetic CUDA repository."""
    use_local_path(
        repository_ctx,
        local_cuda_path,
        ["include", "lib", "bin", "nvvm"],
    )

def _use_local_cudnn_path(repository_ctx, local_cudnn_path):
    # buildifier: disable=function-docstring-args
    """ Creates symlinks and initializes hermetic CUDNN repository."""
    use_local_path(repository_ctx, local_cudnn_path, ["include", "lib"])

def _download_redistribution(repository_ctx, arch_key, path_prefix):
    (url, sha256) = repository_ctx.attr.url_dict[arch_key]

    # If url is not relative, then appending prefix is not needed.
    if not (url.startswith("http") or url.startswith("file:///")):
        url = path_prefix + url
    archive_name = get_archive_name(url)
    file_name = _get_file_name(url)

    print("Downloading and extracting {}".format(url))  # buildifier: disable=print
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

def _get_platform_architecture(repository_ctx):
    target_arch = get_env_var(repository_ctx, "CUDA_REDIST_TARGET_PLATFORM")

    # We use NVCC compiler as the host compiler.
    if target_arch and repository_ctx.name != "cuda_nvcc":
        if target_arch in OS_ARCH_DICT.keys():
            host_arch = target_arch
        else:
            fail(
                "Unsupported architecture: {arch}, use one of {supported}".format(
                    arch = target_arch,
                    supported = OS_ARCH_DICT.keys(),
                ),
            )
    else:
        host_arch = repository_ctx.os.arch

    if host_arch == "aarch64":
        uname_result = repository_ctx.execute(["uname", "-a"]).stdout
        if TEGRA in uname_result:
            return "{}-{}".format(TEGRA, host_arch)
    return host_arch

def _use_downloaded_cuda_redistribution(repository_ctx):
    # buildifier: disable=function-docstring-args
    """ Downloads CUDA redistribution and initializes hermetic CUDA repository."""
    major_version = ""
    cuda_version = (get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    if not cuda_version:
        # If no CUDA version is found, comment out all cc_import targets.
        create_dummy_build_file(repository_ctx)
        _create_cuda_version_file(repository_ctx, {})
        create_version_file(repository_ctx, major_version)
        return

    if len(repository_ctx.attr.url_dict) == 0:
        print("{} is not found in redistributions list.".format(
            repository_ctx.name,
        ))  # buildifier: disable=print
        create_dummy_build_file(repository_ctx)
        _create_cuda_version_file(repository_ctx, {})
        create_version_file(repository_ctx, major_version)
        return

    # Download archive only when GPU config is used.
    arch_key = OS_ARCH_DICT[_get_platform_architecture(repository_ctx)]
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
    _download_redistribution(
        repository_ctx,
        arch_key,
        repository_ctx.attr.cuda_redist_path_prefix,
    )
    lib_name_to_version_dict = get_lib_name_to_version_dict(repository_ctx)
    major_version = get_major_library_version(repository_ctx, lib_name_to_version_dict)
    create_build_file(
        repository_ctx,
        lib_name_to_version_dict,
        major_version,
    )
    _create_libcuda_symlinks(
        repository_ctx,
        lib_name_to_version_dict,
    )
    _create_cuda_header_symlinks(repository_ctx)
    _create_cuda_version_file(repository_ctx, lib_name_to_version_dict)
    create_version_file(repository_ctx, major_version)

def _cuda_repo_impl(repository_ctx):
    local_cuda_path = get_env_var(repository_ctx, "LOCAL_CUDA_PATH")
    if local_cuda_path:
        _use_local_cuda_path(repository_ctx, local_cuda_path)
    else:
        _use_downloaded_cuda_redistribution(repository_ctx)

cuda_repo = repository_rule(
    implementation = _cuda_repo_impl,
    attrs = {
        "url_dict": attr.string_list_dict(mandatory = True),
        "versions": attr.string_list(mandatory = True),
        "build_templates": attr.label_list(mandatory = True),
        "override_strip_prefix": attr.string(),
        "cuda_redist_path_prefix": attr.string(),
    },
    environ = [
        "HERMETIC_CUDA_VERSION",
        "TF_CUDA_VERSION",
        "LOCAL_CUDA_PATH",
        "CUDA_REDIST_TARGET_PLATFORM",
    ],
)

def _use_downloaded_cudnn_redistribution(repository_ctx):
    # buildifier: disable=function-docstring-args
    """ Downloads CUDNN redistribution and initializes hermetic CUDNN repository."""
    cudnn_version = None
    major_version = ""
    cudnn_version = (get_env_var(repository_ctx, "HERMETIC_CUDNN_VERSION") or
                     get_env_var(repository_ctx, "TF_CUDNN_VERSION"))
    cuda_version = (get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    if not cudnn_version:
        # If no CUDNN version is found, comment out cc_import targets.
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
    arch_key = OS_ARCH_DICT[_get_platform_architecture(repository_ctx)]
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

    _download_redistribution(
        repository_ctx,
        arch_key,
        repository_ctx.attr.cudnn_redist_path_prefix,
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

def _cudnn_repo_impl(repository_ctx):
    local_cudnn_path = get_env_var(repository_ctx, "LOCAL_CUDNN_PATH")
    if local_cudnn_path:
        _use_local_cudnn_path(repository_ctx, local_cudnn_path)
    else:
        _use_downloaded_cudnn_redistribution(repository_ctx)

cudnn_repo = repository_rule(
    implementation = _cudnn_repo_impl,
    attrs = {
        "url_dict": attr.string_list_dict(mandatory = True),
        "versions": attr.string_list(mandatory = True),
        "build_templates": attr.label_list(mandatory = True),
        "override_strip_prefix": attr.string(),
        "cudnn_redist_path_prefix": attr.string(),
    },
    environ = [
        "HERMETIC_CUDNN_VERSION",
        "TF_CUDNN_VERSION",
        "HERMETIC_CUDA_VERSION",
        "TF_CUDA_VERSION",
        "LOCAL_CUDNN_PATH",
        "CUDA_REDIST_TARGET_PLATFORM",
    ],
)

def _get_redistribution_urls(dist_info):
    url_dict = {}
    for arch in _REDIST_ARCH_DICT.keys():
        arch_key = arch
        if arch_key == "linux-aarch64" and arch_key not in dist_info:
            arch_key = "linux-sbsa"
        if arch_key not in dist_info:
            continue
        if "relative_path" in dist_info[arch_key]:
            url_dict[_REDIST_ARCH_DICT[arch]] = [
                dist_info[arch_key]["relative_path"],
                dist_info[arch_key].get("sha256", ""),
            ]
            continue

        if "full_path" in dist_info[arch_key]:
            url_dict[_REDIST_ARCH_DICT[arch]] = [
                dist_info[arch_key]["full_path"],
                dist_info[arch_key].get("sha256", ""),
            ]
            continue

        for cuda_version, data in dist_info[arch_key].items():
            # CUDNN JSON might contain paths for each CUDA version.
            path_key = "relative_path"
            if path_key not in data.keys():
                path_key = "full_path"
            url_dict["{cuda_version}_{arch}".format(
                cuda_version = cuda_version,
                arch = _REDIST_ARCH_DICT[arch],
            )] = [data[path_key], data.get("sha256", "")]
    return url_dict

def get_version_and_template_lists(version_to_template):
    # buildifier: disable=function-docstring-return
    # buildifier: disable=function-docstring-args
    """Returns lists of versions and templates provided in the dict."""
    template_to_version_map = {}
    for version, template in version_to_template.items():
        if template not in template_to_version_map.keys():
            template_to_version_map[template] = [version]
        else:
            template_to_version_map[template].append(version)
    version_list = []
    template_list = []
    for template, versions in template_to_version_map.items():
        version_list.append(",".join(versions))
        template_list.append(Label(template))
    return (version_list, template_list)

def cudnn_redist_init_repository(
        cudnn_redistributions,
        cudnn_redist_path_prefix = CUDNN_REDIST_PATH_PREFIX,
        redist_versions_to_build_templates = REDIST_VERSIONS_TO_BUILD_TEMPLATES):
    # buildifier: disable=function-docstring-args
    """Initializes CUDNN repository."""
    if "cudnn" in cudnn_redistributions.keys():
        url_dict = _get_redistribution_urls(cudnn_redistributions["cudnn"])
    else:
        url_dict = {}
    repo_data = redist_versions_to_build_templates["cudnn"]
    versions, templates = get_version_and_template_lists(
        repo_data["version_to_template"],
    )
    cudnn_repo(
        name = repo_data["repo_name"],
        versions = versions,
        build_templates = templates,
        url_dict = url_dict,
        cudnn_redist_path_prefix = cudnn_redist_path_prefix,
    )

def cuda_redist_init_repositories(
        cuda_redistributions,
        cuda_redist_path_prefix = CUDA_REDIST_PATH_PREFIX,
        redist_versions_to_build_templates = REDIST_VERSIONS_TO_BUILD_TEMPLATES):
    # buildifier: disable=function-docstring-args
    """Initializes CUDA repositories."""
    for redist_name, _ in redist_versions_to_build_templates.items():
        if redist_name in ["cudnn", "cuda_nccl"]:
            continue
        if redist_name in cuda_redistributions.keys():
            url_dict = _get_redistribution_urls(cuda_redistributions[redist_name])
        else:
            url_dict = {}
        repo_data = redist_versions_to_build_templates[redist_name]
        versions, templates = get_version_and_template_lists(
            repo_data["version_to_template"],
        )
        cuda_repo(
            name = repo_data["repo_name"],
            versions = versions,
            build_templates = templates,
            url_dict = url_dict,
            cuda_redist_path_prefix = cuda_redist_path_prefix,
        )
