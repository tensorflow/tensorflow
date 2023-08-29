# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Build macros for C++ headers."""

def builtin_ops_visibility_allowlist():
    """Returns a list of packages that can depend on :builtin_ops."""
    return []

def verifier_visibility_allowlist():
    """Returns a list of packages that can depend on :verifier."""
    return []

def delegate_registry_visibility_allowlist():
    """Returns a list of packages that can depend on delegate_registry.h."""
    return []

def macros_visibility_allowlist():
    """Returns a list of packages that can depend on macros.h."""
    return []
