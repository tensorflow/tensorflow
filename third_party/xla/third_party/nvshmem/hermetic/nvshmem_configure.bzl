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
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "bool_setting")
bool_flag(
    name = "include_nvshmem_libs",
    build_setting_default = False,
)

config_setting(
    name = "nvshmem_libs",
    flag_values = {":include_nvshmem_libs": "True"},
)

bool_setting(
    name = "true_setting",
    visibility = ["//visibility:private"],
    build_setting_default = True,
)

config_setting(
    name = "nvshmem_tools",
    flag_values = {":true_setting": "True"},
)
"""

NVSHMEM_DISABLED_BUILD_CONTENT = """
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "bool_setting")

bool_setting(
    name = "true_setting",
    visibility = ["//visibility:private"],
    build_setting_default = True,
)

config_setting(
    name = "nvshmem_tools",
    flag_values = {":true_setting": "False"},
)

config_setting(
    name = "nvshmem_libs",
    flag_values = {":true_setting": "False"},
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
