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
// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text -verify-diagnostics %s | FileCheck %s

// CHECK: HloModule check_imported_configs, {{.*}} replica_count=2, num_partitions=4
module @check_imported_configs attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 2 : i32} {
  func.func @main(%arg0: tensor<1xf32>) -> (tensor<1xf32>) {
    return %arg0 : tensor<1xf32>
  }
}