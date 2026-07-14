// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

func.func @test(%ch: !tfrt.chain, %arg0: !corert.tensorhandle, %arg1_th: !corert.tensorhandle) {
  %cpu = corert.get_op_handler %ch "cpu"
  %0 = corert.executeop(%cpu) "tf.Relu"(%arg0) { T = f32 } : 1
  %arg1 = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %arg1_th {_tfrt_cost = 1 : i64, device = "/CPU:0"} : (!corert.tensorhandle) -> (!tfrt_fallback.tf_tensor)
  %1 = tfrt_fallback_async.executeop key(0) cost(100) device("/CPU:0") "tf.Relu"(%arg1) { T = f32 } : 1
  tfrt.return
}
