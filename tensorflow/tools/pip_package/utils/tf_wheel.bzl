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
6) `--xla_aot` - paths to files that should be in xla_aot directory. 
"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load(
    "@python_version_repo//:py_version.bzl",
    "HERMETIC_PYTHON_VERSION",
    "MACOSX_DEPLOYMENT_TARGET",
    "WHEEL_COLLAB",
    "WHEEL_NAME",
)
load("//tensorflow:tensorflow.bzl", "VERSION")

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

def _get_full_wheel_name(platform_name, platform_tag):
    python_version = HERMETIC_PYTHON_VERSION.replace(".", "")
    return "{wheel_name}-{wheel_version}-SNAPSHOT-cp{python_version}-cp{python_version}-{wheel_platform_tag}.whl".format(
        wheel_name = WHEEL_NAME,
        wheel_version = VERSION,
        python_version = python_version,
        wheel_platform_tag = _get_wheel_platform_name(platform_name, platform_tag),
    )

def _tf_wheel_impl(ctx):
    include_cuda_libs = ctx.attr.include_cuda_libs[BuildSettingInfo].value
    override_include_cuda_libs = ctx.attr.override_include_cuda_libs[BuildSettingInfo].value
    if include_cuda_libs and not override_include_cuda_libs:
        fail("TF wheel shouldn't be built with CUDA dependencies." +
             " Please provide `--config=cuda_wheel` for bazel build command." +
             " If you absolutely need to add CUDA dependencies, provide" +
             " `--@local_config_cuda//cuda:override_include_cuda_libs=true`.")
    executable = ctx.executable.wheel_binary

    full_wheel_name = _get_full_wheel_name(
        platform_name = ctx.attr.platform_name,
        platform_tag = ctx.attr.platform_tag,
    )
    wheel_dir_name = "wheel_house"
    output_file = ctx.actions.declare_file("{wheel_dir}/{wheel_name}".format(
        wheel_dir = wheel_dir_name,
        wheel_name = full_wheel_name,
    ))
    wheel_dir = output_file.path[:output_file.path.rfind("/")]
    check_wheel_compliance = (ctx.attr.platform_name == "linux" and
                              ctx.attr.verify_wheel_compliance[BuildSettingInfo].value and
                              ctx.attr.linux_wheel_compliance_tag)
    args = [
        "--project-name {}".format(WHEEL_NAME),
        "--platform {}".format(_get_wheel_platform_name(
            ctx.attr.platform_name,
            ctx.attr.platform_tag,
        )),
        "--collab {}".format(str(WHEEL_COLLAB)),
        "--output-name {}".format(wheel_dir),
        "--version {}".format(VERSION),
    ]

    headers = ctx.files.headers[:]
    for f in headers:
        args.append("--headers {}".format(f.path))

    xla_aot = ctx.files.xla_aot_compiled[:]
    for f in xla_aot:
        args.append("--xla_aot {}".format(f.path))

    srcs = []
    for src in ctx.attr.source_files:
        for f in src.files.to_list():
            srcs.append(f)
            args.append("--srcs {}".format(f.path))

    args_as_string = ""
    for arg in args:
        args_as_string += arg + " "
    command = executable.path + " " + args_as_string + """\n
    mv {wheel_dir}/*.whl {wheel_file}
    """.format(wheel_dir = wheel_dir, wheel_file = output_file.path)
    ctx.actions.run_shell(
        inputs = srcs + headers + xla_aot,
        command = command,
        outputs = [output_file],
        tools = [executable],
    )
    compliance_verification_log = None
    if check_wheel_compliance:
        compliance_verification_log = ctx.actions.declare_file("compliance_verification.log")
        args = ctx.actions.args()
        args.add("--wheel_path", output_file.path)
        args.add("--compliance-tag", ctx.attr.linux_wheel_compliance_tag)
        args.add("--compliance-verification-log-path", compliance_verification_log.path)
        ctx.actions.run(
            arguments = [args],
            inputs = [output_file],
            outputs = [compliance_verification_log],
            executable = ctx.executable.verify_wheel_compliance_binary,
        )

    verification_output = [compliance_verification_log] if compliance_verification_log else []
    return [DefaultInfo(files = depset(direct = [output_file] + verification_output))]

tf_wheel = rule(
    attrs = {
        "source_files": attr.label_list(allow_files = True),
        "headers": attr.label_list(allow_files = True),
        "xla_aot_compiled": attr.label_list(allow_files = True),
        "wheel_binary": attr.label(
            default = Label("//tensorflow/tools/pip_package:build_pip_package_py"),
            executable = True,
            cfg = "exec",
        ),
        "verify_wheel_compliance_binary": attr.label(
            default = Label("@local_tsl//third_party/py:verify_wheel_compliance_py"),
            executable = True,
            cfg = "exec",
        ),
        "include_cuda_libs": attr.label(default = Label("@local_config_cuda//cuda:include_cuda_libs")),
        "override_include_cuda_libs": attr.label(default = Label("@local_config_cuda//cuda:override_include_cuda_libs")),
        "platform_tag": attr.string(mandatory = True),
        "platform_name": attr.string(mandatory = True),
        "verify_wheel_compliance": attr.label(default = Label("@local_tsl//third_party/py:wheel_compliance")),
        "linux_wheel_compliance_tag": attr.string(),
    },
    implementation = _tf_wheel_impl,
)

def tf_wheel_dep():
    return ["@pypi_{}//:pkg".format(WHEEL_NAME)]
