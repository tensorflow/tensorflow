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

"""Common utilities and argument lists for Python rules wrappers."""

# Attributes unsupported in OSS rules_python and should be stripped.
_UNSUPPORTED_ARGS = [
    "strict_deps",
    "lazy_imports",
    "flaky_test_attempts",
    "linking_mode",
]

def filter_kwargs(kwargs):
    """Filters kwargs to remove unsupported internal attributes."""
    return {k: v for k, v in kwargs.items() if k not in _UNSUPPORTED_ARGS}
