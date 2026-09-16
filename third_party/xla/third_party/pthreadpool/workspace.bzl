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

"""pthreadpool is a portable and efficient thread pool implementation."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "pthreadpool",
        sha256 = "9b9fb1179b71021c0c048504eab636424c58e9c8374404754ece6d7cd90f26d4",
        strip_prefix = "pthreadpool-15a6644ba1c45f1acc16ac1e883efc3e56c6bed2",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/15a6644ba1c45f1acc16ac1e883efc3e56c6bed2.zip"),
    )
