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

"""Loads the xprof library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo(**kwargs):
    """Loads the xprof library, used by TF."""
    tf_http_archive(
        name = "org_xprof",
        sha256 = "4a9c4401c106f3a5dfb5eb481dadf614f567a6e7927e138f2cbe4afaaeed3fd8",
        strip_prefix = "xprof-01b4072213efa05e26b7e3e18f10f5a5a7a13975",
        urls = tf_mirror_urls("https://github.com/openxla/xprof/archive/01b4072213efa05e26b7e3e18f10f5a5a7a13975.zip"),
        **kwargs
    )
