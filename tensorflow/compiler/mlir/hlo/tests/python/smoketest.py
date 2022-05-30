# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Simple smoketest for the Python API."""

# pylint: disable=wildcard-import,undefined-variable

from mlir.dialects.chlo import *
from mlir.dialects.mhlo import *
from mlir.ir import *

ASM = """
func.func @dynamicBroadcast(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
"""

with Context() as context:
  register_chlo_dialect(context)
  register_mhlo_dialect(context)

  m = Module.parse(ASM)
  assert m is not None
  add_op = m.body.operations[0].regions[0].blocks[0].operations[0]
  assert add_op is not None
  print("MHLO Python bindings work")
