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
// RUN: xla-translate --stablehlo-to-hlo %s | xla-translate --hlo-to-stablehlo -mlir-print-debuginfo | FileCheck %s

// Debug locations must survive StableHLO -> HLO -> StableHLO: the op type and
// op name wrappers, the traceback (exported as stack frames) including source
// ranges, and a plain file position with or without a range.

module @jit_f {
  // CHECK-LABEL: func.func @main
  func.func public @main(%arg0: tensor<i32> loc("x")) -> tensor<i32> {
    // CHECK: stablehlo.add {{.*}} loc(#[[ADD:.*]])
    %0 = stablehlo.add %arg0, %arg0 : tensor<i32> loc(#add)
    // CHECK: stablehlo.multiply {{.*}} loc(#[[MUL:.*]])
    %1 = stablehlo.multiply %0, %0 : tensor<i32> loc(#mul)
    // CHECK: stablehlo.subtract {{.*}} loc(#[[SUB:.*]])
    %2 = stablehlo.subtract %1, %0 : tensor<i32> loc(#sub)
    return %2 : tensor<i32>
  }
}
// CHECK-DAG: #[[F_POS:.*]] = loc("src.py":5:9 to :14)
#f_pos = loc("src.py":5:9 to :14)
// CHECK-DAG: #[[MODULE_POS:.*]] = loc("src.py":12:6 to 13:2)
#module_pos = loc("src.py":12:6 to 13:2)
// CHECK-DAG: #[[F:.*]] = loc("f"(#[[F_POS]]))
#f = loc("f"(#f_pos))
// CHECK-DAG: #[[MODULE:.*]] = loc("<module>"(#[[MODULE_POS]]))
#module = loc("<module>"(#module_pos))
// CHECK-DAG: #[[TRACEBACK:.*]] = loc(callsite(#[[F]] at #[[MODULE]]))
#traceback = loc(callsite(#f at #module))
// CHECK-DAG: #[[ADD_NAME:.*]] = loc("jit(f)/add"(#[[TRACEBACK]]))
#add_name = loc("jit(f)/add"(#traceback))
// CHECK-DAG: #[[ADD]] = loc("add:"(#[[ADD_NAME]]))
#add = loc("add:"(#add_name))
// CHECK-DAG: #[[MUL_NAME:.*]] = loc("jit(f)/mul")
#mul_name = loc("jit(f)/mul")
// CHECK-DAG: #[[MUL_POS:.*]] = loc("src.py":7:3 to :21)
#mul_pos = loc("src.py":7:3 to :21)
// CHECK-DAG: #[[MUL]] = loc(fused[#[[MUL_NAME]], #[[MUL_POS]]])
#mul = loc(fused[#mul_name, #mul_pos])
// CHECK-DAG: #[[SUB_NAME:.*]] = loc("jit(f)/sub")
#sub_name = loc("jit(f)/sub")
// CHECK-DAG: #[[SUB_POS:.*]] = loc("src.py":9:5)
#sub_pos = loc("src.py":9:5)
// CHECK-DAG: #[[SUB]] = loc(fused[#[[SUB_NAME]], #[[SUB_POS]]])
#sub = loc(fused[#sub_name, #sub_pos])
