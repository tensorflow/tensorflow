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
"""Smoke test of functions in StableHLO Portable APIs."""

from tensorflow.compiler.mlir.stablehlo import stablehlo


def smoketest():
  """Test StableHLO Portable APIs."""
  assert isinstance(stablehlo.get_api_version(), int)
  assembly = """
    module @jit_f_jax.0 {
      func.func public @main(%arg0: tensor<ui32>) -> tensor<i1> {
        %0 = stablehlo.constant dense<1> : tensor<ui32>
        %1 = "stablehlo.compare"(%arg0, %0) {compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
        return %1 : tensor<i1>
      }
    }
  """
  target = stablehlo.get_current_version()
  artifact = stablehlo.serialize_portable_artifact_str(assembly, target)
  deserialized = stablehlo.deserialize_portable_artifact_str(artifact)
  rountrip = stablehlo.serialize_portable_artifact_str(deserialized, target)
  assert artifact == rountrip


if __name__ == "__main__":
  smoketest()
