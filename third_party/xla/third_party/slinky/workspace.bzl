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

"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "066d6f62a2a6d8eeff6ea75c88aa1d3dc63b754342083e139534133416e9a979",
        strip_prefix = "slinky-713283a0e7c9968e15b33df80feba7337cb19de1",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/713283a0e7c9968e15b33df80feba7337cb19de1.zip"),
    )
