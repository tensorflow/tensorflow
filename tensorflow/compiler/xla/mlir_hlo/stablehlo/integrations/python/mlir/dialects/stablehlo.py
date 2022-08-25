# pylint: disable=missing-module-docstring
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
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

# pylint: disable=wildcard-import,relative-beyond-top-level,g-import-not-at-top
from ._stablehlo_ops_gen import *


def register_dialect(context, load=True):
  from .._mlir_libs import _stablehlo
  _stablehlo.register_dialect(context, load=load)
