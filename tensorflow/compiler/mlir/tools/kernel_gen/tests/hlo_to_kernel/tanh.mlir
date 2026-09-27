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
// RUN: hlo_to_kernel --input=%s --output=%t --unroll_factors=4 --tile_sizes=256 --arch=sm_70

func.func @tanh(%arg0: tensor<*xf32>) -> tensor<*xf32> attributes {tf_entry} {
  %0 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %1 = shape.num_elements %0 : tensor<?xindex> -> index
  %from_elements = tensor.from_elements %1 : tensor<1xindex>
  %2 = tensor.reshape %arg0(%from_elements) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  %3 = mhlo.tanh %2 : tensor<?xf32>
  %4 = tensor.reshape %3(%0) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
  return %4 : tensor<*xf32>
}
