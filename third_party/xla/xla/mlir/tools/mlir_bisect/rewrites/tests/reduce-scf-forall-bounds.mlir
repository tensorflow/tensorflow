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
// RUN: mlir-bisect %s --debug-strategy=ReduceScfForallBounds | FileCheck %s

func.func @main() -> tensor<8xindex> {
  %init = tensor.empty() : tensor<8xindex>
  %iota = scf.forall (%i) = (0) to (8) step (1)
      shared_outs (%init_ = %init) -> (tensor<8xindex>) {
    %tensor = tensor.from_elements %i : tensor<1xindex>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tensor into %init_[%i] [1] [1]
        : tensor<1xindex> into tensor<8xindex>
    }
  }
  func.return %iota : tensor<8xindex>
}
// CHECK: func @main()
// CHECK:   scf.forall ({{.*}}) in (7)
