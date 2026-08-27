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

"""DLPack is a protocol for sharing arrays between deep learning frameworks."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "dlpack",
        strip_prefix = "dlpack-77aafa4d3b0f80feffce9ad4c718dd26751ee0e4",
        sha256 = "c32e9389f4cb079a3d1fdf0077c257650d00f824df4c079c9468b29fcafcfec7",
        urls = tf_mirror_urls("https://github.com/dmlc/dlpack/archive/77aafa4d3b0f80feffce9ad4c718dd26751ee0e4.tar.gz"),
        build_file = "//third_party/dlpack:dlpack.BUILD",
    )
