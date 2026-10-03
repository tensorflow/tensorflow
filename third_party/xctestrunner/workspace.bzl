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

"""Loads the xctestrunner library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "xctestrunner",
        strip_prefix = "xctestrunner-4c5709da9444eae6bba2425734b8654635bed0a6",
        sha256 = "e5d4c53c3965ae943fb08ccd7df0efd75590213fce5052388f23fad81a649f5a",
        urls = tf_mirror_urls("https://github.com/google/xctestrunner/archive/4c5709da9444eae6bba2425734b8654635bed0a6.tar.gz"),
    )
