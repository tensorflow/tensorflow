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
# ==============================================================================

# Platform-specific build configurations.
"""
TF profiler build macros for use in OSS.
"""

load(
    "@rules_ml_toolchain//py/rules_pywrap:pywrap.default.bzl",
    "use_pywrap_rules",
)
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl:tsl.bzl", "cc_header_only_library")

visibility(DEFAULT_LOAD_VISIBILITY)

def tf_profiler_alias(target_dir, name):
    return target_dir + "oss:" + name

def tf_profiler_pybind_cc_library_wrapper(name, actual, **kwargs):
    """Wrapper for cc_library used by tf_python_pybind_extension_opensource.

    This wrapper ensures that cc libraries headers are made available to pybind
    code, without creating ODR violations in the dynamically linked case.  The
    symbols in these deps symbols should be linked to, and exported by, the core
    pywrap_tensorflow_internal.so
    """
    if use_pywrap_rules():
        native.alias(name = name, actual = actual, **kwargs)
    else:
        cc_header_only_library(name = name, deps = [actual], **kwargs)
