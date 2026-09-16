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
// RUN: kernel-gen-opt %s --parallel-loops-to-sequential | FileCheck %s

// CHECK-LABEL: @parallel_loop
func.func @parallel_loop(%lb_0 : index, %lb_1 : index,
                     %ub_0 : index, %ub_1 : index,
                     %s_0 : index, %s_1 : index,
                     %buf: memref<?x?xindex>) {
  scf.parallel (%i0, %i1) = (%lb_0, %lb_1) to (%ub_0, %ub_1) step (%s_0, %s_1) {
    %sum_elem = arith.addi %i0, %i1 : index
    memref.store %sum_elem, %buf[%i0, %i1] : memref<?x?xindex>
  }
  func.return
}
// CHECK: scf.for [[I_0:%.*]] = [[LB_0:%.*]] to [[UB_0:%.*]] step [[S_0:%.*]]
// CHECK:   scf.for [[I_1:%.*]] = [[LB_1:%.*]] to [[UB_1:%.*]] step [[S_1:%.*]]
// CHECK:     [[SUM:%.*]] = arith.addi [[I_0]], [[I_1]] : index
// CHECK:     memref.store [[SUM]], {{%.*}}{{\[}}[[I_0]], [[I_1]]] : memref<?x?xindex>
