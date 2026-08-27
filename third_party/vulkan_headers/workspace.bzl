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

"""Loads Vulkan-Headers, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "vulkan_headers",
        # LINT.IfChange
        strip_prefix = "Vulkan-Headers-32c07c0c5334aea069e518206d75e002ccd85389",
        sha256 = "602aedcc4c6057473d0f7fee1bcc3aa01bf191371b2b5bbca949cebc03cf393a",
        link_files = {
            "//third_party/vulkan_headers:tensorflow/vulkan_hpp_dispatch_loader_dynamic.cc": "tensorflow/vulkan_hpp_dispatch_loader_dynamic.cc",
        },
        urls = tf_mirror_urls("https://github.com/KhronosGroup/Vulkan-Headers/archive/32c07c0c5334aea069e518206d75e002ccd85389.tar.gz"),
        # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/vulkan_headers.cmake)
        build_file = "//third_party/vulkan_headers:vulkan_headers.BUILD",
    )
