# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rule to build custom wheel.

It parses prvoided inputs and then calls `build_pip_package_py` binary with following args:
1) `--project-name` - name to be passed to setup.py file. It will define name of the wheel.
Should be set via --repo_env=WHEEL_NAME=tensorflow_cpu.
2) `--collab` - whether this is a collaborator build.
3) `--version` - tensorflow version.
4) `--headers` - paths to header file.
5) `--srcs` - paths to source files
6) `--dests` - json file with source to destination mappings for files whose original
location does not match its destination in packaged wheel; if the destination is an
empty string the source file will be ignored.
7) `--xla_aot` - paths to files that should be in xla_aot directory. 
"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load(
    "@python_version_repo//:py_version.bzl",
    "HERMETIC_PYTHON_VERSION",
    "MACOSX_DEPLOYMENT_TARGET",
    "WHEEL_COLLAB",
    "WHEEL_NAME",
)
load(
    "//tensorflow:tf_version.bzl",
    "TF_VERSION",
    "TF_WHEEL_VERSION_SUFFIX",
)

def _get_wheel_platform_name(platform_name, platform_tag):
    macos_platform_version = "{}_".format(MACOSX_DEPLOYMENT_TARGET.replace(".", "_")) if MACOSX_DEPLOYMENT_TARGET else ""
    tag = platform_tag
    if platform_tag == "x86_64" and platform_name == "win":
        tag = "amd64"
    if platform_tag == "arm64" and platform_name == "linux":
        tag = "aarch64"
    return "{platform_name}_{platform_version}{platform_tag}".format(
        platform_name = platform_name,
        platform_tag = tag,
        platform_version = macos_platform_version,
    )

def _get_full_wheel_name(
        platform_name,
        platform_tag,
        wheel_version):
    python_version = HERMETIC_PYTHON_VERSION.replace(".", "")
    return "{wheel_name}-{wheel_version}-cp{python_version}-cp{python_version}-{wheel_platform_tag}.whl".format(
        wheel_name = WHEEL_NAME,
        wheel_version = wheel_version,
        python_version = python_version,
        wheel_platform_tag = _get_wheel_platform_name(
            platform_name,
            platform_tag,
        ),
    )

def _is_dest_file(basename, dest_files_suffixes):
    for suffix in dest_files_suffixes:
        if basename.endswith(suffix):
            return True
    return False

def _tf_wheel_impl(ctx):
    include_cuda_libs = ctx.attr.include_cuda_libs[BuildSettingInfo].value
    override_include_cuda_libs = ctx.attr.override_include_cuda_libs[BuildSettingInfo].value
    include_nvshmem_libs = ctx.attr.include_nvshmem_libs[BuildSettingInfo].value
    override_include_nvshmem_libs = ctx.attr.override_include_nvshmem_libs[BuildSettingInfo].value
    if include_cuda_libs and not override_include_cuda_libs:
        fail("TF wheel shouldn't be built with CUDA dependencies." +
             " Please provide `--config=cuda_wheel` for bazel build command." +
             " If you absolutely need to add CUDA dependencies, provide" +
             " `--@local_config_cuda//cuda:override_include_cuda_libs=true`.")
    if include_nvshmem_libs and not override_include_nvshmem_libs:
        fail("TF wheel shouldn't be built directly against the NVSHMEM libraries." +
             " Please provide `--config=cuda_wheel` for bazel build command." +
             " If you absolutely need to build links directly against the NVSHMEM libraries," +
             " `provide --@local_config_nvshmem//:override_include_nvshmem_libs=true`.")
    executable = ctx.executable.wheel_binary

    full_wheel_version = (TF_VERSION + TF_WHEEL_VERSION_SUFFIX)
    full_wheel_name = _get_full_wheel_name(
        platform_name = ctx.attr.platform_name,
        platform_tag = ctx.attr.platform_tag,
        wheel_version = full_wheel_version,
    )
    wheel_dir_name = "wheel_house"
    output_file = ctx.actions.declare_file("{wheel_dir}/{wheel_name}".format(
        wheel_dir = wheel_dir_name,
        wheel_name = full_wheel_name,
    ))
    wheel_dir = output_file.path[:output_file.path.rfind("/")]
    args = ctx.actions.args()
    args.add("--project-name", WHEEL_NAME)
    args.add("--platform", _get_wheel_platform_name(
        ctx.attr.platform_name,
        ctx.attr.platform_tag,
    ))
    args.add("--collab", str(WHEEL_COLLAB))
    args.add("--output-name", wheel_dir)
    args.add("--version", TF_VERSION)

    headers = ctx.files.headers[:]
    for f in headers:
        args.add("--headers=%s" % (f.path))

    xla_aot = ctx.files.xla_aot_compiled[:]
    for f in xla_aot:
        args.add("--xla_aot=%s" % (f.path))

    srcs = []
    for src in ctx.attr.source_files:
        for f in src.files.to_list():
            srcs.append(f)
            if _is_dest_file(f.basename, ctx.attr.dest_files_suffixes):
                args.add("--dests=%s" % (f.path))
            else:
                args.add("--srcs=%s" % (f.path))

    args.set_param_file_format("flag_per_line")
    args.use_param_file("@%s", use_always = False)
    ctx.actions.run(
        arguments = [args],
        inputs = srcs + headers + xla_aot,
        outputs = [output_file],
        executable = executable,
        use_default_shell_env = True,
    )
    return [DefaultInfo(files = depset(direct = [output_file]))]

tf_wheel = rule(
    attrs = {
        "source_files": attr.label_list(allow_files = True),
        "dest_files_suffixes": attr.string_list(default = ["_wheel_locations.json"]),
        "headers": attr.label_list(allow_files = True),
        "xla_aot_compiled": attr.label_list(allow_files = True),
        "wheel_binary": attr.label(
            default = Label("//tensorflow/tools/pip_package:build_pip_package_py"),
            executable = True,
            cfg = "exec",
        ),
        "include_cuda_libs": attr.label(default = Label("@local_config_cuda//cuda:include_cuda_libs")),
        "override_include_cuda_libs": attr.label(default = Label("@local_config_cuda//cuda:override_include_cuda_libs")),
        "include_nvshmem_libs": attr.label(default = Label("@local_config_nvshmem//:include_nvshmem_libs")),
        "override_include_nvshmem_libs": attr.label(default = Label("@local_config_nvshmem//:override_include_nvshmem_libs")),
        "platform_tag": attr.string(mandatory = True),
        "platform_name": attr.string(mandatory = True),
    },
    implementation = _tf_wheel_impl,
)

def tf_wheel_dep():
    return ["@pypi_{}//:pkg".format(WHEEL_NAME)]
