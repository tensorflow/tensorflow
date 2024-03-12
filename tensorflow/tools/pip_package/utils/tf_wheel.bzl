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

load("@python_version_repo//:py_version.bzl", "WHEEL_COLLAB", "WHEEL_NAME")
load("//tensorflow:tensorflow.bzl", "VERSION")

def _tf_wheel_impl(ctx):
    executable = ctx.executable.wheel_binary

    output = ctx.actions.declare_directory("wheel_house")
    args = ctx.actions.args()
    args.add("--project-name", WHEEL_NAME)
    args.add("--collab", str(WHEEL_COLLAB))
    args.add("--output-name", output.path)
    args.add("--version", VERSION)

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
            args.add("--srcs=%s" % (f.path))

    args.set_param_file_format("flag_per_line")
    args.use_param_file("@%s", use_always = False)
    ctx.actions.run(
        arguments = [args],
        inputs = srcs + headers + xla_aot,
        outputs = [output],
        executable = executable,
    )
    return [DefaultInfo(files = depset(direct = [output]))]

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
    },
    implementation = _tf_wheel_impl,
)
