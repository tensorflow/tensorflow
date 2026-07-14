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
# ============================================================================
"""Wrapper for py_extension in OSS using native.cc_binary."""

def py_extension(name, srcs, deps = [], copts = [], linkopts = [], **kwargs):
    # Remove features or other attributes unsupported by cc_binary
    kwargs.pop("features", None)

    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = copts + ["-fexceptions", "-fPIC"],
        linkshared = True,
        linkopts = linkopts,
        deps = deps + [
            "@rules_python//python/cc:current_py_cc_headers",
        ],
        **kwargs
    )
