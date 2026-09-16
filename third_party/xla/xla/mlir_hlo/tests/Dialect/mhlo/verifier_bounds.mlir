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
// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file

// expected-error@+1 {{Bounds length is 1, expected to be equal to rank(2) of the tensor}}
func.func @incorrect_bounds_length(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [3]>>) -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [3]>> {
  func.return %arg0 : tensor<?x?xf32, #mhlo.type_extensions<bounds = [3]>>
}

// -----

// expected-error@+1 {{Static dimension 0 cannot have a bound, use ShapedType::kDynamic to indicate a missing bound}}
func.func @static_dim_with_bound(%arg0: tensor<3xf32, #mhlo.type_extensions<bounds = [3]>>) -> tensor<?xf32, #mhlo.type_extensions<bounds = [3]>> {
  func.return %arg0 : tensor<?xf32, #mhlo.type_extensions<bounds = [3]>>
}
