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

"""Wrapper for py_binary rule."""

load("@rules_python//python:defs.bzl", _py_binary = "py_binary")
load("//third_party/rules_python/python:common.bzl", "filter_kwargs")

def py_binary(**kwargs):
    """Wrapper for py_binary that strictly filters supported attributes."""
    _py_binary(**filter_kwargs(kwargs))
