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

"""Repository rule for hermetic NVSHMEM configuration. """

load("@nvidia_nvshmem//:version.bzl", _nvshmem_version = "VERSION")
load(
    "//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "enable_cuda",
    "use_cuda_redistributions",
)
load(
    "//third_party/remote_config:common.bzl",
    "get_cpu_value",
)

NVSHMEM_ENABLED_BUILD_CONTENT = """
package(default_visibility = ["//visibility:public"])

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "bool_setting")

# This set of flags and config_settings is needed to enable NVSHMEM dependencies
# separately from CUDA dependencies. The reason is that NVSHMEM libraries
# require GLIBC 2.28 and above, which we don't have on RBE runners yet.
# TODO(ybaturina): Remove this once GLIBC 2.28 is available on RBE.
bool_flag(
    name = "include_nvshmem_libs",
    build_setting_default = False,
)

config_setting(
    name = "nvshmem_libs",
    flag_values = {":include_nvshmem_libs": "True"},
)

bool_flag(
    name = "override_include_nvshmem_libs",
    build_setting_default = False,
)

config_setting(
    name = "overrided_nvshmem_libs",
    flag_values = {":override_include_nvshmem_libs": "True"},
)

alias(
    name = "nvshmem_tools",
    actual = "@local_config_cuda//:is_cuda_enabled",
)

selects.config_setting_group(
    name = "any_nvshmem_libs",
    match_any = [
        ":nvshmem_libs",
        ":overrided_nvshmem_libs",
    ],
)

selects.config_setting_group(
    name = "nvshmem_tools_and_libs",
    match_all = [
        ":any_nvshmem_libs",
        ":nvshmem_tools",
    ],
)
"""

NVSHMEM_DISABLED_BUILD_CONTENT = """
package(default_visibility = ["//visibility:public"])

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "bool_setting")

bool_setting(
    name = "true_setting",
    build_setting_default = True,
)

bool_flag(
    name = "include_nvshmem_libs",
    build_setting_default = False,
)

config_setting(
    name = "nvshmem_tools",
    flag_values = {":true_setting": "False"},
)

config_setting(
    name = "nvshmem_libs",
    flag_values = {":true_setting": "False"},
)

bool_flag(
    name = "override_include_nvshmem_libs",
    build_setting_default = False,
)

config_setting(
    name = "overrided_nvshmem_libs",
    flag_values = {":true_setting": "False"},
)

selects.config_setting_group(
    name = "any_nvshmem_libs",
    match_any = [
        ":nvshmem_libs",
        ":overrided_nvshmem_libs"
    ],
)

selects.config_setting_group(
    name = "nvshmem_tools_and_libs",
    match_all = [
        ":any_nvshmem_libs",
        ":nvshmem_tools"
    ],
)
"""

def _nvshmem_autoconf_impl(repository_ctx):
    if (not enable_cuda(repository_ctx) or
        get_cpu_value(repository_ctx) != "Linux"):
        repository_ctx.file("BUILD", NVSHMEM_DISABLED_BUILD_CONTENT)
        if use_cuda_redistributions(repository_ctx):
            repository_ctx.file(
                "nvshmem_config.h",
                "#define NVSHMEM_VERSION \"%s\"" % _nvshmem_version,
            )
        else:
            repository_ctx.file("nvshmem_config.h", "#define NVSHMEM_VERSION \"\"")
    else:
        repository_ctx.file(
            "nvshmem_config.h",
            "#define NVSHMEM_VERSION \"%s\"" % _nvshmem_version,
        )
        repository_ctx.file("BUILD", NVSHMEM_ENABLED_BUILD_CONTENT)

nvshmem_configure = repository_rule(
    implementation = _nvshmem_autoconf_impl,
)
"""Downloads and configures the hermetic NVSHMEM configuration.

Add the following to your WORKSPACE file:

```python
nvshmem_configure(name = "local_config_nvshmem")
```

Args:
  name: A unique name for this workspace rule.
"""  # buildifier: disable=no-effect
