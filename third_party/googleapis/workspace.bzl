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

"""Loads the googleapis library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "com_google_googleapis",
        build_file = "//third_party/googleapis:googleapis.BUILD",
        sha256 = "249d83abc5d50bf372c35c49d77f900bff022b2c21eb73aa8da1458b6ac401fc",
        strip_prefix = "googleapis-6b3fdcea8bc5398be4e7e9930c693f0ea09316a0",
        urls = tf_mirror_urls("https://github.com/googleapis/googleapis/archive/6b3fdcea8bc5398be4e7e9930c693f0ea09316a0.tar.gz"),
    )
