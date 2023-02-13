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
"""External-only build rules for delegate plugins."""

def delegate_plugin_visibility_allowlist():
    """Returns a list of packages that can depend on delegate_plugin."""
    return []

def gpu_plugin_visibility_allowlist():
    """Returns a list of packages that can depend on gpu_plugin."""
    return []

def xnnpack_plugin_visibility_allowlist():
    """Returns a list of packages that can depend on xnnpack_plugin."""
    return []

def stable_delegate_visibility_allowlist():
    """Returns a list of packages that can depend on stable_delegate."""
    return []
