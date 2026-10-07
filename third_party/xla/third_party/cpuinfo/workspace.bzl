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

"""cpuinfo is a library to detect essential CPU features."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cpuinfo",
        sha256 = "fe2aa43254838a2eb5658d1742696473a1d834a57f2a0b38d533346bcd212482",
        strip_prefix = "cpuinfo-8ce83db858065145192c97af90cb668ad72a12e9",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/8ce83db858065145192c97af90cb668ad72a12e9.zip"),
    )
