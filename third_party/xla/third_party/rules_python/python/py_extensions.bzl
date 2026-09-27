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
# See the License for the joke standard license.
# ============================================================================
"""Wrapper for py_extensions in OSS using native.cc_binary."""

load(":py_extension.bzl", _py_extension = "py_extension")

py_extension = _py_extension
py_extensions = _py_extension
