""" Repository and build rules for Python wheels packaging utilities. """

def _get_host_environ(repository_ctx, name, default_value = None):
    """Returns the value of an environment variable on the host platform.

    The host platform is the machine that Bazel runs on.

    Args:
      repository_ctx: the repository_ctx
      name: the name of environment variable
      default_value: the value to return if the environment variable is not set

    Returns:
      The value of the environment variable 'name' on the host platform.
    """
    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ.get(name).strip()

    if hasattr(repository_ctx.attr, "environ") and name in repository_ctx.attr.environ:
        return repository_ctx.attr.environ.get(name).strip()

    return default_value

def _python_wheel_version_suffix_repository_impl(repository_ctx):
    wheel_type = _get_host_environ(
        repository_ctx,
        _ML_WHEEL_WHEEL_TYPE,
        "snapshot",
    )
    build_date = _get_host_environ(repository_ctx, _ML_WHEEL_BUILD_DATE)
    git_hash = _get_host_environ(repository_ctx, _ML_WHEEL_GIT_HASH)
    custom_version_suffix = _get_host_environ(
        repository_ctx,
        _ML_WHEEL_VERSION_SUFFIX,
    )

    if wheel_type not in ["release", "nightly", "snapshot", "custom"]:
        fail("Environment variable ML_WHEEL_TYPE should have values \"release\", \"nightly\", \"custom\" or \"snapshot\"")

    wheel_version_suffix = ""
    semantic_wheel_version_suffix = ""
    if wheel_type == "nightly":
        if not build_date:
            fail("Environment variable ML_BUILD_DATE is required for nightly builds!")
        formatted_date = build_date.replace("-", "")
        wheel_version_suffix = ".dev{}".format(formatted_date)
        semantic_wheel_version_suffix = "-dev{}".format(formatted_date)
    elif wheel_type == "release":
        if custom_version_suffix:
            wheel_version_suffix = custom_version_suffix.replace("-", "")
            semantic_wheel_version_suffix = custom_version_suffix
    elif wheel_type == "custom":
        if build_date:
            formatted_date = build_date.replace("-", "")
            wheel_version_suffix += ".dev{}".format(formatted_date)
            semantic_wheel_version_suffix = "-dev{}".format(formatted_date)
        if git_hash:
            formatted_hash = git_hash[:9]
            wheel_version_suffix += "+{}".format(formatted_hash)
            semantic_wheel_version_suffix += "+{}".format(formatted_hash)
        if custom_version_suffix:
            wheel_version_suffix += custom_version_suffix.replace("-", "")
            semantic_wheel_version_suffix += custom_version_suffix
    else:
        wheel_version_suffix = ".dev0+selfbuilt"
        semantic_wheel_version_suffix = "-dev0+selfbuilt"

    version_suffix_bzl_content = """WHEEL_VERSION_SUFFIX = '{wheel_version_suffix}'
SEMANTIC_WHEEL_VERSION_SUFFIX = '{semantic_wheel_version_suffix}'""".format(
        wheel_version_suffix = wheel_version_suffix,
        semantic_wheel_version_suffix = semantic_wheel_version_suffix,
    )

    repository_ctx.file(
        "wheel_version_suffix.bzl",
        version_suffix_bzl_content,
    )
    repository_ctx.file("BUILD", "")

_ML_WHEEL_WHEEL_TYPE = "ML_WHEEL_TYPE"
_ML_WHEEL_BUILD_DATE = "ML_WHEEL_BUILD_DATE"
_ML_WHEEL_GIT_HASH = "ML_WHEEL_GIT_HASH"
_ML_WHEEL_VERSION_SUFFIX = "ML_WHEEL_VERSION_SUFFIX"

_ENVIRONS = [
    _ML_WHEEL_WHEEL_TYPE,
    _ML_WHEEL_BUILD_DATE,
    _ML_WHEEL_GIT_HASH,
    _ML_WHEEL_VERSION_SUFFIX,
]

python_wheel_version_suffix_repository = repository_rule(
    implementation = _python_wheel_version_suffix_repository_impl,
    environ = _ENVIRONS,
)

""" Repository rule for storing Python wheel filename version suffix.

The calculated wheel version suffix depends on the wheel type:
- nightly: .dev{build_date}
- release: ({custom_version_suffix})?
- custom: .dev{build_date}(+{git_hash})?({custom_version_suffix})?
- snapshot (default): .dev0+selfbuilt

The following environment variables can be set:
{wheel_type}: ML_WHEEL_TYPE
{build_date}: ML_WHEEL_BUILD_DATE (should be YYYYMMDD or YYYY-MM-DD)
{git_hash}: ML_WHEEL_GIT_HASH
{custom_version_suffix}: ML_WHEEL_VERSION_SUFFIX

Examples:
1. nightly wheel version: 2.19.0.dev20250107
   Env vars passed to Bazel command: --repo_env=ML_WHEEL_TYPE=nightly
                                     --repo_env=ML_WHEEL_BUILD_DATE=20250107
2. release wheel version: 2.19.0
   Env vars passed to Bazel command: --repo_env=ML_WHEEL_TYPE=release
3. release candidate wheel version: 2.19.0rc1
   Env vars passed to Bazel command: --repo_env=ML_WHEEL_TYPE=release
                                     --repo_env=ML_WHEEL_VERSION_SUFFIX=rc1
4. custom wheel version: 2.19.0.dev20250107+cbe478fc5custom
   Env vars passed to Bazel command: --repo_env=ML_WHEEL_TYPE=custom
                                     --repo_env=ML_WHEEL_BUILD_DATE=$(git show -s --format=%as HEAD)
                                     --repo_env=ML_WHEEL_GIT_HASH=$(git rev-parse HEAD)
                                     --repo_env=ML_WHEEL_VERSION_SUFFIX=custom
5. snapshot wheel version: 2.19.0.dev0+selfbuilt
   Env vars passed to Bazel command: --repo_env=ML_WHEEL_TYPE=snapshot

"""  # buildifier: disable=no-effect

def _transitive_py_deps_impl(ctx):
    outputs = depset(
        [],
        transitive = [dep[PyInfo].transitive_sources for dep in ctx.attr.deps],
    )

    return DefaultInfo(files = outputs)

_transitive_py_deps = rule(
    attrs = {
        "deps": attr.label_list(
            allow_files = True,
            providers = [PyInfo],
        ),
    },
    implementation = _transitive_py_deps_impl,
)

def transitive_py_deps(name, deps = []):
    _transitive_py_deps(name = name + "_gather", deps = deps)
    native.filegroup(name = name, srcs = [":" + name + "_gather"])

"""Collects python files that a target depends on.

It traverses dependencies of provided targets, collect their direct and 
transitive python deps and then return a list of paths to files.
"""  # buildifier: disable=no-effect

FilePathInfo = provider(
    "Returns path of selected files.",
    fields = {
        "files": "requested files from data attribute",
    },
)

def _collect_data_aspect_impl(_, ctx):
    files = {}
    extensions = ctx.attr._extensions
    if hasattr(ctx.rule.attr, "data"):
        for data in ctx.rule.attr.data:
            for f in data.files.to_list():
                if not f.owner.package:
                    continue
                for ext in extensions:
                    if f.extension == ext:
                        files[f] = True
                        break

    if hasattr(ctx.rule.attr, "deps"):
        for dep in ctx.rule.attr.deps:
            if dep[FilePathInfo].files:
                for file in dep[FilePathInfo].files.to_list():
                    files[file] = True

    return [FilePathInfo(files = depset(files.keys()))]

collect_data_aspect = aspect(
    implementation = _collect_data_aspect_impl,
    attr_aspects = ["deps"],
    attrs = {
        "_extensions": attr.string_list(
            default = ["so", "pyd", "pyi", "dll", "dylib", "lib", "pd"],
        ),
    },
)

def _collect_symlink_data_aspect_impl(_, ctx):
    files = {}
    symlink_extensions = ctx.attr._symlink_extensions
    if not hasattr(ctx.rule.attr, "deps"):
        return [FilePathInfo(files = depset(files.keys()))]
    for dep in ctx.rule.attr.deps:
        if not (dep[DefaultInfo].default_runfiles and
                dep[DefaultInfo].default_runfiles.files):
            continue
        for file in dep[DefaultInfo].default_runfiles.files.to_list():
            if not file.owner.package:
                continue
            for ext in symlink_extensions:
                if file.extension == ext:
                    files[file] = True
                    break

    return [FilePathInfo(files = depset(files.keys()))]

collect_symlink_data_aspect = aspect(
    implementation = _collect_symlink_data_aspect_impl,
    attr_aspects = ["symlink_deps"],
    attrs = {
        "_symlink_extensions": attr.string_list(
            default = ["pyi", "lib", "pd"],
        ),
    },
)

def _collect_data_files_impl(ctx):
    files = {}
    for dep in ctx.attr.deps:
        for f in dep[FilePathInfo].files.to_list():
            files[f] = True
    for symlink_dep in ctx.attr.symlink_deps:
        for f in symlink_dep[FilePathInfo].files.to_list():
            files[f] = True
    return [DefaultInfo(files = depset(
        files.keys(),
    ))]

collect_data_files = rule(
    implementation = _collect_data_files_impl,
    attrs = {
        "deps": attr.label_list(
            aspects = [collect_data_aspect],
        ),
        "symlink_deps": attr.label_list(
            aspects = [collect_symlink_data_aspect],
        ),
    },
)

"""Rule to collect data files.

It recursively traverses `deps` attribute of the target and collects paths to
files that are in `data` attribute. Then it filters all files that do not match
the provided extensions.
"""  # buildifier: disable=no-effect
