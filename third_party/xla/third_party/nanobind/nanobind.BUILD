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
# ==============================================================================

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

exports_files([
    "src/stubgen.py",
])

cc_library(
    name = "nanobind",
    srcs = glob(
        [
            "src/*.cpp",
        ],
        exclude = [
            "src/nb_backend.cpp",
            "src/nb_combined.cpp",
        ],
    ),
    copts = ["-fexceptions"],
    # On Linux/macOS, NB_SHARED=1 gives nanobind symbols default visibility so
    # they can be shared across extensions. On Windows, NB_SHARED=1 causes
    # downstream headers (where NB_BUILD is not defined) to declare nanobind
    # functions as __declspec(dllimport) (__imp_*), which fails to link when
    # nanobind's object files are linked directly into the binary.
    defines = select({
        "@platforms//os:windows": [],
        "//conditions:default": ["NB_SHARED=1"],
    }) + select({
        "@rules_python//python/config_settings:is_py_freethreaded": [
            "NB_FREE_THREADED=1",
        ],
        "//conditions:default": [],
    }),
    includes = ["include"],
    local_defines = ["NB_BUILD=1"],
    textual_hdrs = glob(
        [
            "include/**/*.h",
            "include/**/*.inl",
            "src/*.h",
        ],
    ),
    deps = [
        "@robin_map",
        "@xla//third_party/python_runtime:headers",
    ],
)
