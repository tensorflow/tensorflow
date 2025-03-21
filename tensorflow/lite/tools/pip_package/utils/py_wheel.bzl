# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Rule to build a python wheel generically.

This rule is used to build a python wheel from a list of source files. It takes a list of source
files, a setup.py file, and a version string as input. It then uses a python script,
wheel_builder.py, to generate the wheel file. The wheel builder binary is responsible for preparing
the build environment and calling the setuptools command to generate the wheel file.
"""

load(
    "@python_version_repo//:py_version.bzl",
    "HERMETIC_PYTHON_VERSION",
)

def _get_full_wheel_name(wheel_name, version, platform_name):
    python_version = HERMETIC_PYTHON_VERSION.replace(".", "")
    wheel_version = version.replace("-dev", ".dev").replace("-", "")
    return "{wheel_name}-{wheel_version}-cp{python_version}-cp{python_version}-{wheel_platform_tag}.whl".format(
        wheel_name = wheel_name,
        wheel_version = wheel_version,
        python_version = python_version,
        wheel_platform_tag = platform_name,
    )

def _py_wheel_impl(ctx):
    executable = ctx.executable.wheel_binary
    filelist_lists = [src.files.to_list() for src in ctx.attr.srcs]
    filelist = [f for filelist in filelist_lists for f in filelist]
    wheel_name = _get_full_wheel_name("ai_edge_litert", ctx.attr.version, ctx.attr.platform_name)
    output_file = ctx.actions.declare_file("dist/{wheel_name}".format(wheel_name = wheel_name))

    args = ctx.actions.args()
    args.add("--setup_py", ctx.file.setup_py.path)
    args.add("--output", output_file.dirname)
    args.add("--version", ctx.attr.version)

    for f in filelist:
        args.add("--src", f.path)

    if ctx.attr.platform_name:
        args.add("--platform", ctx.attr.platform_name)

    args.set_param_file_format("flag_per_line")
    args.use_param_file("@%s", use_always = False)

    ctx.actions.run(
        mnemonic = "WheelBuilder",
        arguments = [args],
        inputs = filelist + [ctx.file.setup_py],
        outputs = [output_file],
        executable = executable,
    )
    return [DefaultInfo(files = depset(direct = [output_file]))]

py_wheel = rule(
    implementation = _py_wheel_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
        ),
        "pyproject": attr.label(
            allow_single_file = [".toml"],
        ),
        "setup_py": attr.label(
            allow_single_file = [".py"],
            mandatory = True,
        ),
        "platform_name": attr.string(),
        "version": attr.string(mandatory = True),
        "wheel_binary": attr.label(
            default = Label("//tensorflow/lite/tools/pip_package/utils:wheel_builder"),
            executable = True,
            cfg = "exec",
        ),
    },
)
