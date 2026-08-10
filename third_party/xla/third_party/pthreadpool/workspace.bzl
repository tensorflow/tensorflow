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
        sha256 = "5ab4e8f63e3dcf62048360c216532bdf62f00dc204883a52d91230402f0feb6a",
        strip_prefix = "pthreadpool-02460584c6092e527c8b89f7df4de143d70e801f",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/02460584c6092e527c8b89f7df4de143d70e801f.zip"),
    )
