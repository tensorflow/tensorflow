# Copyright 2026 The OpenXLA Authors. All rights reserved.
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

"""Repository rules and utilities for archives with quilt-style patch series."""

def read_patch_series(ctx, series_spec):
    """Reads one or more quilt-style series files and returns a list of patch Path objects.

    Args:
      ctx: The repository rule context.
      series_spec: A string label or list of string labels of series files.

    Returns:
      A list of Path objects for patches in series order.
    """
    if type(series_spec) == "string":
        series_spec = [series_spec]

    patches = []
    for series_label in series_spec:
        if not series_label:
            continue
        series_file = ctx.path(Label(series_label))
        if not series_file.exists:
            fail("Patch series file does not exist: %s" % series_label)
        for line in ctx.read(series_file).splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patch_path = series_file.dirname
                for part in line.split("/"):
                    if part and part != ".":
                        patch_path = patch_path.get_child(part)
                patches.append(patch_path)
    return patches

def _get_link_dict(ctx, link_files, build_file):
    link_dict = {ctx.path(v): ctx.path(Label(k)) for k, v in link_files.items()}
    if build_file:
        # Use BUILD.bazel because it takes precedence over BUILD.
        link_dict[ctx.path("BUILD.bazel")] = ctx.path(Label(build_file))
    return link_dict

def _quilt_http_archive_impl(ctx):
    link_dict = _get_link_dict(ctx, ctx.attr.link_files, ctx.attr.build_file)

    # Resolve labels before download_and_extract to prevent unnecessary re-downloads.
    # Borrowed from tf_http_archive (https://github.com/bazelbuild/bazel/issues/10515).
    patch_files = [ctx.path(Label(p)) for p in ctx.attr.patch_file if p]
    if ctx.attr.series:
        patch_files.extend(read_patch_series(ctx, ctx.attr.series))

    ctx.download_and_extract(
        url = ctx.attr.urls,
        sha256 = ctx.attr.sha256,
        type = ctx.attr.type,
        stripPrefix = ctx.attr.strip_prefix,
    )

    for patch_file in patch_files:
        ctx.patch(patch_file, strip = ctx.attr.patch_strip)

    if hasattr(ctx.attr, "patch_cmds") and ctx.attr.patch_cmds:
        for cmd in ctx.attr.patch_cmds:
            res = ctx.execute(["bash", "-c", cmd])
            if res.return_code != 0:
                fail("patch_cmd failed: %s\n%s" % (cmd, res.stderr))

    for dst, src in link_dict.items():
        ctx.delete(dst)
        ctx.symlink(src, dst)

_quilt_http_archive = repository_rule(
    implementation = _quilt_http_archive_impl,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "series": attr.string_list(),
        "patch_file": attr.string_list(),
        "patch_strip": attr.int(default = 1),
        "patch_cmds": attr.string_list(),
        "build_file": attr.string(),
        "link_files": attr.string_dict(),
    },
)

def quilt_http_archive(name, sha256, urls, series = [], **kwargs):
    """Downloads and creates Bazel repos for dependencies with quilt-style patch series.

    Args:
      name: A unique name for this repository.
      sha256: The SHA-256 of the archive file.
      urls: A list of URLs where the archive can be downloaded.
      series: Label or list of labels of patch series files.
      **kwargs: Additional keyword arguments passed to the repository rule.
    """
    if len(urls) < 2:
        fail("quilt_http_archive(urls) must have redundant URLs.")

    if not any([mirror in urls[0] for mirror in (
        "mirror.tensorflow.org",
        "mirror.bazel.build",
        "storage.googleapis.com",
    )]):
        fail("The first entry of quilt_http_archive(urls) must be a mirror " +
             "URL, preferably mirror.tensorflow.org. Even if you don't have " +
             "permission to mirror the file, please put the correctly " +
             "formatted mirror URL there anyway, because someone will come " +
             "along shortly thereafter and mirror the file.")

    if native.existing_rule(name):
        return

    if type(series) == "string":
        series = [series]

    _quilt_http_archive(
        name = name,
        sha256 = sha256,
        urls = urls,
        series = series,
        **kwargs
    )
