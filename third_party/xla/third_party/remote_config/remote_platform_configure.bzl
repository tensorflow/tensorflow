# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
# =============================================================================

"""Repository rule to create a platform for a docker image to be used with RBE."""

def _remote_platform_configure_impl(repository_ctx):
    platform = repository_ctx.attr.platform
    if platform == "local":
        os = repository_ctx.os.name.lower()
        if os.startswith("windows"):
            platform = "windows"
        elif os.startswith("mac os"):
            platform = "osx"
        else:
            platform = "linux"

    cpu = "x86_64"
    machine_type = repository_ctx.execute(["bash", "-c", "echo $MACHTYPE"]).stdout
    if (machine_type.startswith("ppc") or
        machine_type.startswith("powerpc")):
        cpu = "ppc"
    elif machine_type.startswith("s390x"):
        cpu = "s390x"
    elif machine_type.startswith("aarch64"):
        cpu = "aarch64"
    elif machine_type.startswith("arm64"):
        cpu = "aarch64"
    elif machine_type.startswith("arm"):
        cpu = "arm"
    elif machine_type.startswith("mips64"):
        cpu = "mips64"
    elif machine_type.startswith("riscv64"):
        cpu = "riscv64"

    exec_properties = repository_ctx.attr.platform_exec_properties

    serialized_exec_properties = "{"
    for k, v in exec_properties.items():
        serialized_exec_properties += "\"%s\" : \"%s\"," % (k, v)
    serialized_exec_properties += "}"

    repository_ctx.template(
        "BUILD",
        Label("//third_party/remote_config:BUILD.tpl"),
        {
            "%{platform}": platform,
            "%{exec_properties}": serialized_exec_properties,
            "%{cpu}": cpu,
        },
    )

remote_platform_configure = repository_rule(
    implementation = _remote_platform_configure_impl,
    attrs = {
        "platform_exec_properties": attr.string_dict(mandatory = True),
        "platform": attr.string(default = "linux", values = ["linux", "windows", "local"]),
    },
)
