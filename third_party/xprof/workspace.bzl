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
        sha256 = "d964d6a8101236f85f7e4a15a7e835ef674c61cc647ca9de3e7a5a1a119bbe56",
        strip_prefix = "xprof-29b42d9060811ea1cf5464aa4368853db8329737",
        urls = tf_mirror_urls("https://github.com/openxla/xprof/archive/29b42d9060811ea1cf5464aa4368853db8329737.zip"),
        **kwargs
    )
