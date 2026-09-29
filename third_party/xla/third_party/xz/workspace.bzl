# Copyright 2020 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "tf_http_archive",
    "tf_mirror_urls",
)

def repo():
    maybe(
        tf_http_archive,
        # The repo name is the same as the one used in llvm_project to avoid the conflict with `xz`
        # repository in rules_ml_toolchain.
        name = "llvm_xz",
        strip_prefix = "xz-5.8.3",
        urls = tf_mirror_urls("https://github.com/tukaani-project/xz/releases/download/v5.8.3/xz-5.8.3.tar.gz"),
        sha256 = "3d3a1b973af218114f4f889bbaa2f4c037deaae0c8e815eec381c3d546b974a0",
        patch_file = [
            "//third_party/xz/patches:remove_have_config.diff",
        ],
        build_file = "//third_party/xz:xz.BUILD.bazel",
    )
