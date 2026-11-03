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
        sha256 = "8ca25be2a8750d47a5bea7e2b85e035ebed1c87729802b310c05ccd7f419f9f0",
        strip_prefix = "slinky-4449dd1499efbe239ab6ee320058450bab59b97b",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/4449dd1499efbe239ab6ee320058450bab59b97b.zip"),
    )
