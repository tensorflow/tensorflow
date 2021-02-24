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

# Apply a patch_file to the repository root directory.
def _apply_patch(ctx, patch_file):
    ctx.patch(patch_file, strip = 1)

def _maybe_label(label_string):
    return Label(label_string) if label_string else None

def _label_path_dict(ctx, dict):
    return {Label(k): ctx.path(v) for k, v in dict.items()}

def _tf_http_archive(ctx):
    # Construct all labels early on to prevent rule restart. We want the
    # attributes to be strings instead of labels because they refer to files
    # in the TensorFlow repository, not files in repos depending on TensorFlow.
    # See also https://github.com/bazelbuild/bazel/issues/10515.
    patch_file = _maybe_label(ctx.attr.patch_file)
    build_file = _maybe_label(ctx.attr.build_file)
    system_build_file = _maybe_label(ctx.attr.system_build_file)
    system_link_files = _label_path_dict(ctx, ctx.attr.system_link_files)
    additional_build_files = _label_path_dict(ctx, ctx.attr.additional_build_files)

    if len(ctx.attr.urls) < 2 or "mirror.tensorflow.org" not in ctx.attr.urls[0]:
        fail("tf_http_archive(urls) must have redundant URLs. The " +
             "mirror.tensorflow.org URL must be present and it must come first. " +
             "Even if you don't have permission to mirror the file, please " +
             "put the correctly formatted mirror URL there anyway, because " +
             "someone will come along shortly thereafter and mirror the file.")

    use_syslib = _use_system_lib(ctx, ctx.attr.name)

    if not use_syslib:
        ctx.download_and_extract(
            ctx.attr.urls,
            "",
            ctx.attr.sha256,
            ctx.attr.type,
            ctx.attr.strip_prefix,
        )
        if patch_file:
            _apply_patch(ctx, patch_file)

    if use_syslib and system_build_file:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", system_build_file, executable = False)
    elif build_file:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", build_file, executable = False)

    if use_syslib:
        for label, path in system_link_files.items():
            ctx.symlink(label, path)

    for label, path in additional_build_files.items():
        ctx.symlink(label, path)

tf_http_archive = repository_rule(
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "patch_file": attr.string(),
        "build_file": attr.string(),
        "system_build_file": attr.string(),
        "system_link_files": attr.string_dict(),
        "additional_build_files": attr.string_dict(),
    },
    environ = [
        "TF_SYSTEM_LIBS",
    ],
    implementation = _tf_http_archive,
    doc = """Downloads and creates Bazel repos for dependencies.

This is a swappable replacement for both http_archive() and
new_http_archive() that offers some additional features. It also helps
ensure best practices are followed.

File arguments are relative to the TensorFlow repository by default. Dependent
repositories that use this rule should refer to files either with absolute
labels (e.g. '@foo//:bar') or from a label created in their repository (e.g.
'str(Label("//:bar"))').""",
)

def _third_party_http_archive(ctx):
    # Construct all labels early on to prevent rule restart. We want the
    # attributes to be strings instead of labels because they refer to files
    # in the TensorFlow repository, not files in repos depending on TensorFlow.
    # See also https://github.com/bazelbuild/bazel/issues/10515.
    build_file = _maybe_label(ctx.attr.build_file)
    system_build_file = _maybe_label(ctx.attr.system_build_file)
    patch_file = _maybe_label(ctx.attr.patch_file)
    link_files = _label_path_dict(ctx, ctx.attr.link_files)
    system_link_files = _label_path_dict(ctx, ctx.attr.system_link_files)

    if len(ctx.attr.urls) < 2 or "mirror.tensorflow.org" not in ctx.attr.urls[0]:
        fail("tf_http_archive(urls) must have redundant URLs. The " +
             "mirror.tensorflow.org URL must be present and it must come first. " +
             "Even if you don't have permission to mirror the file, please " +
             "put the correctly formatted mirror URL there anyway, because " +
             "someone will come along shortly thereafter and mirror the file.")

    use_syslib = _use_system_lib(ctx, ctx.attr.name)

    # Use "BUILD.bazel" to avoid conflict with third party projects that contain a
    # file or directory called "BUILD"
    buildfile_path = ctx.path("BUILD.bazel")

    if use_syslib:
        if ctx.attr.system_build_file == None:
            fail("Bazel was configured with TF_SYSTEM_LIBS to use a system " +
                 "library for %s, but no system build file for %s was configured. " +
                 "Please add a system_build_file attribute to the repository rule" +
                 "for %s." % (ctx.attr.name, ctx.attr.name, ctx.attr.name))
        ctx.symlink(Label(ctx.attr.system_build_file), buildfile_path)

    else:
        ctx.download_and_extract(
            ctx.attr.urls,
            "",
            ctx.attr.sha256,
            ctx.attr.type,
            ctx.attr.strip_prefix,
        )
        if ctx.attr.patch_file:
            _apply_patch(ctx, Label(ctx.attr.patch_file))
        ctx.symlink(Label(ctx.attr.build_file), buildfile_path)

    link_dict = {}
    if use_syslib:
        link_dict.update(system_link_files)

    for label, path in link_files.items():
        # if syslib and link exists in both, use the system one
        if path not in link_dict.values():
            link_dict[label] = path

    for label, path in link_dict.items():
        ctx.symlink(label, path)

# Downloads and creates Bazel repos for dependencies.
#
# This is an upgrade for tf_http_archive that works with go/tfbr-thirdparty.
#
# For link_files, specify each dict entry as:
# "//path/to/source:file": "localfile"
third_party_http_archive = repository_rule(
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(
            mandatory = True,
            allow_empty = False,
        ),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "build_file": attr.string(mandatory = True),
        "system_build_file": attr.string(),
        "patch_file": attr.string(),
        "link_files": attr.string_dict(),
        "system_link_files": attr.string_dict(),
    },
    environ = ["TF_SYSTEM_LIBS"],
    implementation = _third_party_http_archive,
)
