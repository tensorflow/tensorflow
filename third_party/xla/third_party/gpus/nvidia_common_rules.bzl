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

"""Common rules and functions for hermetic NVIDIA repositories."""

load("//third_party:repo.bzl", "tf_mirror_urls")

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
_SUPPORTED_ARCHIVE_EXTENSIONS = [
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
_TEGRA = "tegra"
_LIB_EXTENSION = ".so."

def get_env_var(repository_ctx, name):
    return repository_ctx.getenv(name)

def get_cuda_version(repository_ctx):
    return (get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
            get_env_var(repository_ctx, "TF_CUDA_VERSION"))

def _get_file_name(url):
    last_slash_index = url.rfind("/")
    return url[last_slash_index + 1:]

def get_archive_name(url):
    # buildifier: disable=function-docstring-return
    # buildifier: disable=function-docstring-args
    """Returns the archive name without extension."""
    filename = _get_file_name(url)
    for extension in _SUPPORTED_ARCHIVE_EXTENSIONS:
        if filename.endswith(extension):
            return filename[:-len(extension)]
    return filename

def _get_lib_name_and_version(path):
    extension_index = path.rfind(_LIB_EXTENSION)
    last_slash_index = path.rfind("/")
    lib_name = path[last_slash_index + 1:extension_index]
    lib_version = path[extension_index + len(_LIB_EXTENSION):]
    return (lib_name, lib_version)

def _get_main_lib_name(repository_ctx):
    if repository_ctx.name == "cuda_driver":
        return "libcuda"
    if repository_ctx.name == "nvidia_nvshmem":
        return "libnvshmem_host"
    else:
        return "lib{}".format(
            _get_common_lib_name(repository_ctx),
        )

def _get_common_lib_name(repository_ctx):
    return repository_ctx.name.split("_")[1].lower()

def _get_libraries_by_redist_name_in_dir(repository_ctx):
    lib_dir_path = repository_ctx.path("lib")
    if not lib_dir_path.exists:
        return []
    common_lib_name = _get_common_lib_name(repository_ctx)
    lib_dir_content = lib_dir_path.readdir()
    return [
        str(f)
        for f in lib_dir_content
        if (_LIB_EXTENSION in str(f) and
            common_lib_name in str(f).lower())
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

def _create_repository_symlinks(repository_ctx):
    for target, link_name in repository_ctx.attr.repository_symlinks.items():
        repository_ctx.symlink(target, link_name)

def create_version_file(repository_ctx, major_lib_version):
    repository_ctx.file(
        "version.bzl",
        "VERSION = \"{}\"".format(major_lib_version),
    )

def use_local_redist_path(repository_ctx, local_redist_path, dirs):
    # buildifier: disable=function-docstring-args
    """Creates repository using local redistribution paths."""
    _create_symlinks(
        repository_ctx,
        local_redist_path,
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
    create_version_file(repository_ctx, major_version)

def _download_redistribution(
        repository_ctx,
        arch_key,
        path_prefix,
        mirrored_tar_path_prefix):
    # buildifier: disable=function-docstring-args
    """Downloads and extracts NVIDIA redistribution."""
    (url, sha256) = repository_ctx.attr.url_dict[arch_key]

    # If url is not relative, then appending prefix is not needed.
    if not (url.startswith("http") or url.startswith("file:///")):
        if url.endswith(".tar"):
            url = mirrored_tar_path_prefix + url
        else:
            url = path_prefix + url
    archive_name = get_archive_name(url)
    file_name = _get_file_name(url)
    urls = [url] if url.endswith(".tar") else tf_mirror_urls(url)

    print("Downloading and extracting {}".format(url))  # buildifier: disable=print
    repository_ctx.download(
        url = urls,
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
    # buildifier: disable=function-docstring-return
    # buildifier: disable=function-docstring-args
    """Returns the platform architecture for the redistribution."""
    target_arch = get_env_var(repository_ctx, repository_ctx.attr.target_arch_env_var)

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
        if _TEGRA in uname_result:
            return "{}-{}".format(_TEGRA, host_arch)
    return host_arch

def _use_downloaded_redistribution(repository_ctx):
    # buildifier: disable=function-docstring-args
    """ Downloads redistribution and initializes hermetic repository."""
    major_version = ""
    redist_version = _get_redist_version(
        repository_ctx,
        repository_ctx.attr.redist_version_env_vars,
    )
    cuda_version = get_cuda_version(repository_ctx)

    if not redist_version:
        # If no toolkit version is found, comment out cc_import targets.
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
            ("{dist_name}: The supported platforms are {supported_platforms}." +
             " Platform {platform} is not supported.")
                .format(
                supported_platforms = repository_ctx.attr.url_dict.keys(),
                platform = arch_key,
                dist_name = repository_ctx.name,
            ),
        )

    _download_redistribution(
        repository_ctx,
        arch_key,
        repository_ctx.attr.redist_path_prefix,
        repository_ctx.attr.mirrored_tar_redist_path_prefix,
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
    _create_repository_symlinks(repository_ctx)
    create_version_file(repository_ctx, major_version)

def _redist_repo_impl(repository_ctx):
    local_redist_path = get_env_var(repository_ctx, repository_ctx.attr.local_path_env_var)
    if local_redist_path:
        use_local_redist_path(repository_ctx, local_redist_path, repository_ctx.attr.local_source_dirs)
    else:
        _use_downloaded_redistribution(repository_ctx)

_redist_repo = repository_rule(
    implementation = _redist_repo_impl,
    attrs = {
        "url_dict": attr.string_list_dict(mandatory = True),
        "versions": attr.string_list(mandatory = True),
        "build_templates": attr.label_list(mandatory = True),
        "override_strip_prefix": attr.string(),
        "redist_path_prefix": attr.string(),
        "mirrored_tar_redist_path_prefix": attr.string(mandatory = False),
        "redist_version_env_vars": attr.string_list(mandatory = True),
        "local_path_env_var": attr.string(mandatory = True),
        "use_tar_file_env_var": attr.string(mandatory = True),
        "target_arch_env_var": attr.string(mandatory = True),
        "local_source_dirs": attr.string_list(mandatory = False),
        "repository_symlinks": attr.label_keyed_string_dict(mandatory = False),
    },
)

def redist_init_repository(
        name,
        versions,
        build_templates,
        url_dict,
        redist_path_prefix,
        mirrored_tar_redist_path_prefix,
        redist_version_env_vars,
        local_path_env_var,
        use_tar_file_env_var,
        target_arch_env_var,
        local_source_dirs,
        override_strip_prefix = "",
        repository_symlinks = {}):
    # buildifier: disable=function-docstring-args
    """Initializes repository for individual NVIDIA redistribution."""
    _redist_repo(
        name = name,
        url_dict = url_dict,
        versions = versions,
        build_templates = build_templates,
        override_strip_prefix = override_strip_prefix,
        redist_path_prefix = redist_path_prefix,
        mirrored_tar_redist_path_prefix = mirrored_tar_redist_path_prefix,
        redist_version_env_vars = redist_version_env_vars,
        local_path_env_var = local_path_env_var,
        use_tar_file_env_var = use_tar_file_env_var,
        target_arch_env_var = target_arch_env_var,
        local_source_dirs = local_source_dirs,
        repository_symlinks = repository_symlinks,
    )

def get_redistribution_urls(dist_info):
    # buildifier: disable=function-docstring-return
    # buildifier: disable=function-docstring-args
    """Returns a dict of redistribution URLs and their SHA256 values."""
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
            # CUDNN and NVSHMEM JSON might contain paths for each CUDA version.
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

def _get_json_file_content(
        repository_ctx,
        url_to_sha256,
        mirrored_tars_url_to_sha256,
        json_file_name,
        mirrored_tars_json_file_name,
        use_tar_file_env_var):
    # buildifier: disable=function-docstring-args
    # buildifier: disable=function-docstring-return
    """ Returns the JSON file content for the NVIDIA redistributions."""
    use_tars = get_env_var(
        repository_ctx,
        use_tar_file_env_var,
    )
    (url, sha256) = url_to_sha256
    if mirrored_tars_url_to_sha256:
        (mirrored_tar_url, mirrored_tar_sha256) = mirrored_tars_url_to_sha256
    else:
        mirrored_tar_url = None
        mirrored_tar_sha256 = None
    json_file = None

    if use_tars and mirrored_tar_url:
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

def _get_redist_version(repository_ctx, redist_version_env_vars):
    redist_version = None
    for redist_version_env_var in redist_version_env_vars:
        redist_version = get_env_var(repository_ctx, redist_version_env_var)
        if redist_version:
            break
    return redist_version

def _redist_json_impl(repository_ctx):
    redist_version = _get_redist_version(
        repository_ctx,
        repository_ctx.attr.redist_version_env_vars,
    )
    local_redist_path = get_env_var(repository_ctx, repository_ctx.attr.local_path_env_var)
    supported_redist_versions = repository_ctx.attr.json_dict.keys()
    if (redist_version and not local_redist_path and
        (redist_version not in supported_redist_versions)):
        fail(
            ("The supported {toolkit_name} versions are {supported_versions}." +
             " Please provide a supported version in {redist_version_env_vars}" +
             " environment variable(s) or add JSON URL for" +
             " {toolkit_name} version={version}.")
                .format(
                toolkit_name = repository_ctx.attr.toolkit_name,
                redist_version_env_vars = repository_ctx.attr.redist_version_env_vars,
                supported_versions = supported_redist_versions,
                version = redist_version,
            ),
        )
    redistributions = "{}"
    if redist_version and not local_redist_path:
        if redist_version in repository_ctx.attr.mirrored_tars_json_dict.keys():
            mirrored_tars_url_to_sha256 = repository_ctx.attr.mirrored_tars_json_dict[redist_version]
        else:
            mirrored_tars_url_to_sha256 = {}
        redistributions = _get_json_file_content(
            repository_ctx,
            url_to_sha256 = repository_ctx.attr.json_dict[redist_version],
            mirrored_tars_url_to_sha256 = mirrored_tars_url_to_sha256,
            json_file_name = "redistrib_{toolkit_name}_{redist_version}.json".format(
                toolkit_name = repository_ctx.attr.toolkit_name,
                redist_version = redist_version,
            ),
            mirrored_tars_json_file_name = "redistrib_{toolkit_name}_{redist_version}_tar.json".format(
                toolkit_name = repository_ctx.attr.toolkit_name,
                redist_version = redist_version,
            ),
            use_tar_file_env_var = repository_ctx.attr.use_tar_file_env_var,
        )

    repository_ctx.file(
        "distributions.bzl",
        "{toolkit_name}_REDISTRIBUTIONS = {redistributions}".format(
            toolkit_name = repository_ctx.attr.toolkit_name,
            redistributions = redistributions,
        ),
    )
    repository_ctx.file(
        "BUILD",
        "",
    )

_redist_json = repository_rule(
    implementation = _redist_json_impl,
    attrs = {
        "toolkit_name": attr.string(mandatory = True),
        "json_dict": attr.string_list_dict(mandatory = True),
        "mirrored_tars_json_dict": attr.string_list_dict(mandatory = True),
        "redist_version_env_vars": attr.string_list(mandatory = True),
        "local_path_env_var": attr.string(mandatory = True),
        "use_tar_file_env_var": attr.string(mandatory = True),
    },
)

def json_init_repository(
        name,
        toolkit_name,
        json_dict,
        mirrored_tars_json_dict,
        redist_version_env_vars,
        local_path_env_var,
        use_tar_file_env_var):
    # buildifier: disable=function-docstring-args
    """Initializes NVIDIA redistribution JSON repository."""
    _redist_json(
        name = name,
        toolkit_name = toolkit_name,
        json_dict = json_dict,
        mirrored_tars_json_dict = mirrored_tars_json_dict,
        redist_version_env_vars = redist_version_env_vars,
        local_path_env_var = local_path_env_var,
        use_tar_file_env_var = use_tar_file_env_var,
    )
