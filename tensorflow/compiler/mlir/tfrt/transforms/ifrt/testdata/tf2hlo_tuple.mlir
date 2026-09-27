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

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main(%arg0: tensor<1x3xf32>, %arg1: tensor<3x1xf32>) -> (tensor<1x1xf32>, tensor<1x3xf32>) attributes {__tpu_compile_metadata_text = "args { dtype: DT_FLOAT kind: PARAMETER } args { dtype: DT_FLOAT kind: PARAMETER } retvals { } retvals { } num_replicas: 1 num_cores_per_replica: 1"} {
    %0 = "tf.MatMul"(%arg0, %arg1): (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    func.return %0, %arg0: tensor<1x1xf32>, tensor<1x3xf32>
  }
}