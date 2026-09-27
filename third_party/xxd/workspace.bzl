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

"""xxd is a tool that converts a binary file into a C/C++ header file containing an array of bytes."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "xxd",
        sha256 = "a5cdcfcfeb13dc4deddcba461d40234dbf47e61941cb7170c9ebe147357bb62d",
        strip_prefix = "vim-9.1.0917/src/xxd",
        urls = tf_mirror_urls("https://github.com/vim/vim/archive/refs/tags/v9.1.0917.tar.gz"),
        build_file = "//third_party/xxd:xxd.BUILD",
    )
