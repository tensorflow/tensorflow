# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""StableHLO Portable Python APIs.

This setup only exports the the StableHLO Portable C++ APIs, which have
signatures that do not rely on MLIR classes.

Exporting all of MLIR Python bindings to TF OSS has high maintenance
implications, especially given the frequency that TF updates the revision of
LLVM used.
"""

# pylint: disable=wildcard-import
from .stablehlo_extension import *
