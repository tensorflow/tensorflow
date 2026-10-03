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

"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_vendored")

# buildifier: disable=function-docstring
# buildifier: disable=unnamed-macro
def workspace():
    # Declares @tsl
    tf_vendored(name = "tsl", path = "third_party/tsl")

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace4 = workspace
