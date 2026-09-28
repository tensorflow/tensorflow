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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --serialize-debug-metadata=true %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir --mlir-print-debuginfo -o - | FileCheck %s
// This test verifies that debug locations are round-trippable.

#loc = loc("<stdin>":0:0)
#loc5 = loc("main"(#loc))
#loc8 = loc("cond_true"(#loc))
#loc10 = loc("cond_false"(#loc))
module @jit_relu attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, tfl._legalize_tfl_variables = true} {
  func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %0 = tfl.less(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1> loc(#loc6)
    // CHECK-DAG: {{.*}} = tfl.less(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1> loc(#[[LESS_LOC:.*]])
    %1 = "tf.If"(%0, %arg0, %arg1) <{else_branch = @cond_false, is_stateless = false, then_branch = @cond_true}> : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32> loc(#loc7)
    // CHECK-DAG: {{.*}} = "tf.If"(%0, %arg0, %arg1) {{.*}} -> tensor<1xf32> loc(#[[IF_LOC:.*]])
    func.return %1 : tensor<1xf32> loc(#loc)
  } loc(#loc5)
  func.func @cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc(#loc16)
    // CHECK-DAG: {{.*}} = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc(#[[ADD_LOC:.*]])
    func.return %0 : tensor<*xf32> loc(#loc)
  } loc(#loc8)
  func.func @cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc(#loc15)
    // CHECK-DAG: {{.*}} = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc(#[[MUL_LOC:.*]])
    func.return %0 : tensor<*xf32> loc(#loc)
  } loc(#loc10)
} loc(#loc)
#loc1 = loc("tfl.less")
#loc2 = loc("tf.If")
#loc3 = loc("<ipython-input-7-340b9abeb7a8>":1:4)
#loc4 = loc("third_party/py/IPython/v3_2_3/core/interactiveshell.py":3066:16)
#loc6 = loc(fused<"tflite.importer_wrapper">[#loc1])
#loc7 = loc(fused<"tflite.importer_wrapper">[#loc2])
#loc9 = loc(callsite(#loc3 at #loc4))
#loc11 = loc(fused<"">[#loc3, #loc4])
#loc12 = loc("jit(relu)/jit(main)/max"(#loc9))
#loc13 = loc("tfl.mul"(#loc11))
#loc14 = loc("jit(relu)/jit(main)/max"(#loc12))
#loc15 = loc(fused<"tflite.importer_wrapper">[#loc13])
#loc16 = loc(fused<"tflite.importer_wrapper">[#loc14])

// CHECK-DAG: #[[LOC1:.*]] = loc("tfl.less")
// CHECK-DAG: #[[LESS_WRAP:.*]] = loc(fused<"tflite.importer_wrapper">[#[[LOC1]]])
// CHECK-DAG: #[[LESS_LOC]] = loc(fused<"tflite.subgraph=0.op=0:LESS">[#[[LESS_WRAP]]])

// CHECK-DAG: #[[LOC2:.*]] = loc("tf.If")
// CHECK-DAG: #[[IF_WRAP:.*]] = loc(fused<"tflite.importer_wrapper">[#[[LOC2]]])
// CHECK-DAG: #[[IF_LOC]] = loc(fused<"tflite.subgraph=0.op=1:IF">[#[[IF_WRAP]]])

// CHECK-DAG: #[[IPY_LOC:.*]] = loc("<ipython-input-7-340b9abeb7a8>":1:4)
// CHECK-DAG: #[[SHELL_LOC:.*]] = loc("third_party/py/IPython/v3_2_3/core/interactiveshell.py":3066:16)
// CHECK-DAG: #[[CALLSITE_LOC:.*]] = loc(callsite(#[[IPY_LOC]] at #[[SHELL_LOC]]))
// CHECK-DAG: #[[MAX_LOC1:.*]] = loc("jit(relu)/jit(main)/max"(#[[CALLSITE_LOC]]))
// CHECK-DAG: #[[MAX_LOC2:.*]] = loc("jit(relu)/jit(main)/max"(#[[MAX_LOC1]]))
// CHECK-DAG: #[[ADD_WRAP:.*]] = loc(fused<"tflite.importer_wrapper">[#[[MAX_LOC2]]])
// CHECK-DAG: #[[ADD_LOC]] = loc(fused<"tflite.subgraph=1.op=0:ADD">[#[[ADD_WRAP]]])

// CHECK-DAG: #[[FUSED_EMPTY:.*]] = loc(fused<"">[#[[IPY_LOC]], #[[SHELL_LOC]]])
// CHECK-DAG: #[[MUL_NAME:.*]] = loc("tfl.mul"(#[[FUSED_EMPTY]]))
// CHECK-DAG: #[[MUL_WRAP:.*]] = loc(fused<"tflite.importer_wrapper">[#[[MUL_NAME]]])
// CHECK-DAG: #[[MUL_LOC]] = loc(fused<"tflite.subgraph=2.op=0:MUL">[#[[MUL_WRAP]]])
