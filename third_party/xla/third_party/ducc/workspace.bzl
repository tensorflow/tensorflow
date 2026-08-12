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

"""Distinctly Useful Code Collection (DUCC) - CPU FFT Module"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    DUCC_COMMIT = "aa46a4c21e440b3d416c16eca3c96df19c74f316"
    DUCC_SHA256 = "077cf4bd0bd7eddaa6649a024285fff96e2662c5e6f2fb6ed5c5771f9de093f3"
    tf_http_archive(
        name = "ducc",
        strip_prefix = "ducc-{commit}".format(commit = DUCC_COMMIT),
        sha256 = DUCC_SHA256,
        urls = tf_mirror_urls("https://gitlab.mpcdf.mpg.de/mtr/ducc/-/archive/{commit}/ducc-{commit}.tar.gz".format(commit = DUCC_COMMIT)),
        build_file = "//third_party/ducc:ducc.BUILD",
        link_files = {
            "//third_party/ducc:ducc0_custom_lowlevel_threading.h": "google/ducc0_custom_lowlevel_threading.h",
            "//third_party/ducc:fft.h": "google/fft.h",
            "//third_party/ducc:fft.cc": "google/fft.cc",
            "//third_party/ducc:threading.cc": "google/threading.cc",
            "//third_party/ducc:threading.h": "google/threading.h",
        },
    )
