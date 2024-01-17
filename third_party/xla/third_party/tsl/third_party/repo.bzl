# Copyright 2017 The TensorFlow Authors. All rights reserved.
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

"""Utilities for defining TensorFlow Bazel dependencies."""

def tf_mirror_urls(url):
    """A helper for generating TF-mirror versions of URLs.

    Given a URL, it returns a list of the TF-mirror cache version of that URL
    and the original URL, suitable for use in `urls` field of `tf_http_archive`.
    """
    if not url.startswith("https://"):
        return [url]
    return [
        "https://storage.googleapis.com/mirror.tensorflow.org/%s" % url[8:],
        url,
    ]

def _get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

# Checks if we should use the system lib instead of the bundled one
def _use_system_lib(ctx, name):
    syslibenv = _get_env_var(ctx, "TF_SYSTEM_LIBS")
    if not syslibenv:
        return False
    return name in [n.strip() for n in syslibenv.split(",")]

def _get_link_dict(ctx, link_files, build_file):
    link_dict = {ctx.path(v): ctx.path(Label(k)) for k, v in link_files.items()}
    if build_file:
        # Use BUILD.bazel because it takes precedence over BUILD.
        link_dict[ctx.path("BUILD.bazel")] = ctx.path(Label(build_file))
    return link_dict

def _tf_http_archive_impl(ctx):
    # Construct all paths early on to prevent rule restart. We want the
    # attributes to be strings instead of labels because they refer to files
    # in the TensorFlow repository, not files in repos depending on TensorFlow.
    # See also https://github.com/bazelbuild/bazel/issues/10515.
    link_dict = _get_link_dict(ctx, ctx.attr.link_files, ctx.attr.build_file)

    # For some reason, we need to "resolve" labels once before the
    # download_and_extract otherwise it'll invalidate and re-download the
    # archive each time.
    # https://github.com/bazelbuild/bazel/issues/10515
    patch_files = ctx.attr.patch_file
    for patch_file in patch_files:
        if patch_file:
            ctx.path(Label(patch_file))

    if _use_system_lib(ctx, ctx.attr.name):
        link_dict.update(_get_link_dict(
            ctx = ctx,
            link_files = ctx.attr.system_link_files,
            build_file = ctx.attr.system_build_file,
        ))
    else:
        ctx.download_and_extract(
            url = ctx.attr.urls,
            sha256 = ctx.attr.sha256,
            type = ctx.attr.type,
            stripPrefix = ctx.attr.strip_prefix,
        )
        if patch_files:
            for patch_file in patch_files:
                patch_file = ctx.path(Label(patch_file)) if patch_file else None
                if patch_file:
                    ctx.patch(patch_file, strip = 1)

    for dst, src in link_dict.items():
        ctx.delete(dst)
        ctx.symlink(src, dst)

_tf_http_archive = repository_rule(
    implementation = _tf_http_archive_impl,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "patch_file": attr.string_list(),
        "build_file": attr.string(),
        "system_build_file": attr.string(),
        "link_files": attr.string_dict(),
        "system_link_files": attr.string_dict(),
    },
    environ = ["TF_SYSTEM_LIBS"],
)

def tf_http_archive(name, sha256, urls, **kwargs):
    """Downloads and creates Bazel repos for dependencies.

    This is a swappable replacement for both http_archive() and
    new_http_archive() that offers some additional features. It also helps
    ensure best practices are followed.

    File arguments are relative to the TensorFlow repository by default. Dependent
    repositories that use this rule should refer to files either with absolute
    labels (e.g. '@foo//:bar') or from a label created in their repository (e.g.
    'str(Label("//:bar"))').
    """
    if len(urls) < 2:
        fail("tf_http_archive(urls) must have redundant URLs.")

    if not any([mirror in urls[0] for mirror in (
        "mirror.tensorflow.org",
        "mirror.bazel.build",
        "storage.googleapis.com",
    )]):
        fail("The first entry of tf_http_archive(urls) must be a mirror " +
             "URL, preferrably mirror.tensorflow.org. Even if you don't have " +
             "permission to mirror the file, please put the correctly " +
             "formatted mirror URL there anyway, because someone will come " +
             "along shortly thereafter and mirror the file.")

    if native.existing_rule(name):
        print("\n\033[1;33mWarning:\033[0m skipping import of repository '" +
              name + "' because it already exists.\n")
        return

    _tf_http_archive(
        name = name,
        sha256 = sha256,
        urls = urls,
        **kwargs
    )

def _tf_vendored_impl(repository_ctx):
    parent_path = repository_ctx.path(repository_ctx.attr.parent).dirname

    # get_child doesn't allow slashes. Yes this is silly. bazel_skylib paths
    # doesn't work with path objects.
    relpath_parts = repository_ctx.attr.relpath.split("/")
    vendored_path = parent_path
    for part in relpath_parts:
        vendored_path = vendored_path.get_child(part)
    repository_ctx.symlink(vendored_path, ".")

tf_vendored = repository_rule(
    implementation = _tf_vendored_impl,
    attrs = {
        "parent": attr.label(default = "//:WORKSPACE"),
        "relpath": attr.string(),
    },
)
