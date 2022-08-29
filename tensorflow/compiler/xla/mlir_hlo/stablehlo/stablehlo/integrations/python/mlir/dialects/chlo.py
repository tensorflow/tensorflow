# pylint: disable=missing-module-docstring
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
from ._chlo_ops_gen import *


def register_dialect(context, load=True):
  from .._mlir_libs import _chlo
  _chlo.register_dialect(context, load=load)


# Backward compatibility with the old way of registering CHLO dialect
def register_chlo_dialect(context, load=True):
  register_dialect(context, load)
