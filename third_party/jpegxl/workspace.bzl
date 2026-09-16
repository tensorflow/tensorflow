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

"""Point to the JPEG XL repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "jpegxl",
        strip_prefix = "libjxl-0.11.1",
        sha256 = "1492dfef8dd6c3036446ac3b340005d92ab92f7d48ee3271b5dac1d36945d3d9",
        urls = tf_mirror_urls("https://github.com/libjxl/libjxl/archive/refs/tags/v0.11.1.tar.gz"),
        build_file = "//third_party/jpegxl:jpegxl.BUILD.bazel",
        patch_file = ["//third_party/jpegxl:external_deps.patch"],
    )
