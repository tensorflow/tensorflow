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

"""Provides the repository macro to import mpitrampoline."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports mpitrampoline."""

    MPITRAMPOLINE_COMMIT = "25efb0f7a4cd00ed82bafb8b1a6285fc50d297ed"
    MPITRAMPOLINE_SHA256 = "5a36656205c472bdb639bffebb0f014523b32dda0c2cbedd9ce7abfc9e879e84"

    tf_http_archive(
        name = "mpitrampoline",
        sha256 = MPITRAMPOLINE_SHA256,
        strip_prefix = "MPItrampoline-{commit}".format(commit = MPITRAMPOLINE_COMMIT),
        urls = tf_mirror_urls("https://github.com/eschnett/mpitrampoline/archive/{commit}.tar.gz".format(commit = MPITRAMPOLINE_COMMIT)),
        patch_file = ["//third_party/mpitrampoline:gen.patch"],
        build_file = "//third_party/mpitrampoline:mpitrampoline.BUILD",
    )
