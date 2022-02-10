# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""MLIR Dialect for mhlo operations."""

# pylint: disable=wildcard-import,relative-beyond-top-level,g-import-not-at-top
from ._mhlo_ops_gen import *
from .._mlir_libs._mlirHlo import *
# pylint: disable=undefined-variable
del register_chlo_dialect
# pylint: enable=undefined-variable
