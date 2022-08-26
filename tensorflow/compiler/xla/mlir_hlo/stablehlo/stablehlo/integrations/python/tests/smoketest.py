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
"""Simple smoketest for the Python API."""

# pylint: disable=wildcard-import,undefined-variable

# TODO(burmako): Uncomment after deleting MLIR-HLO's CHLO.
# from mlir.dialects import chlo
from mlir.dialects import stablehlo
from mlir.ir import *

ASM = """
func.func @test(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // TODO(burmako): Uncomment after deleting MLIR-HLO's CHLO.
  // %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = stablehlo.add %arg1, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = stablehlo.add %0, %0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}
"""

with Context() as context:
  # TODO(burmako): Uncomment after deleting MLIR-HLO's CHLO.
  # chlo.register_dialect(context)
  stablehlo.register_dialect(context)

  m = Module.parse(ASM)
  assert m is not None
  block = m.body.operations[0].regions[0].blocks[0]
  assert block is not None
  assert block.operations[0] is not None
  assert block.operations[1] is not None
  assert block.operations[2] is not None
  print("StableHLO Python bindings seem to work")
