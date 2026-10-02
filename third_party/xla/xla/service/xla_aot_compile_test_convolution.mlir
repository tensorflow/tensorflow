// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

module @foo {
  func.func public @main(%inputs : tensor<1x4x4x2xf32>, %weights : tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32> {
    %res = "mhlo.convolution"(%inputs, %weights) {
          batch_group_count = 1 : i64,
          dimension_numbers = #mhlo.conv<raw
            input_batch_dimension = 0,
            input_feature_dimension = 3,
            input_spatial_dimensions = [1, 2],
            kernel_input_feature_dimension = 2,
            kernel_output_feature_dimension = 3,
            kernel_spatial_dimensions = [0, 1],
            output_batch_dimension = 0,
            output_feature_dimension = 3,
            output_spatial_dimensions = [1, 2]
          >,
          feature_group_count = 1 : i64,
          rhs_dilation = dense<1> : tensor<2xi64>,
          window_strides = dense<1> : tensor<2xi64>} : (tensor<1x4x4x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32>
    return %res : tensor<1x2x3x1xf32>
  }
}