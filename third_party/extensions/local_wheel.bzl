# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Repository rule exposing a locally built wheel from dist/ as a target.

Used by //tensorflow/tools/pip_package:prebuilt_tf_py_import when testing
prebuilt wheels under Bzlmod without patching rules_python.
"""

def _local_wheel_repo_impl(repository_ctx):
    wheel_name = repository_ctx.os.environ.get("WHEEL_NAME", "tensorflow").replace("-", "_")
    dist_dir = repository_ctx.workspace_root.get_child("dist")
    repository_ctx.watch(dist_dir)

    matched_wheel = None
    if dist_dir.exists and dist_dir.is_dir:
        prefix = wheel_name + "-"
        for entry in dist_dir.readdir(watch = "yes"):
            if entry.basename.startswith(prefix) and entry.basename.endswith(".whl"):
                if matched_wheel == None or matched_wheel.basename < entry.basename:
                    matched_wheel = entry

    srcs = []
    if matched_wheel != None:
        repository_ctx.watch(matched_wheel)
        repository_ctx.symlink(matched_wheel, matched_wheel.basename)
        srcs = [matched_wheel.basename]

    repository_ctx.file(
        "BUILD.bazel",
        """package(default_visibility = ["//visibility:public"])

filegroup(
    name = "whl",
    srcs = {srcs},
)
""".format(srcs = repr(srcs)),
    )

local_wheel_repo = repository_rule(
    implementation = _local_wheel_repo_impl,
    environ = ["WHEEL_NAME"],
)
