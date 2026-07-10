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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --serialize-debug-metadata=true %s -o - | flatbuffer_to_string - | FileCheck %s

#loc = loc("<stdin>":0:0)
#loc5 = loc("main"(#loc))
#loc8 = loc("cond_true"(#loc))
#loc10 = loc("cond_false"(#loc))
module @jit_relu attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, tfl._legalize_tfl_variables = true} {
  func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %0 = tfl.less(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1> loc(#loc6)
    %1 = "tf.If"(%0, %arg0, %arg1) <{else_branch = @cond_false, is_stateless = false, then_branch = @cond_true}> : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32> loc(#loc7)
    func.return %1 : tensor<1xf32> loc(#loc)
  } loc(#loc5)
  func.func @cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc(#loc16)
    func.return %0 : tensor<*xf32> loc(#loc)
  } loc(#loc8)
  func.func @cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc(#loc15)
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

// CHECK: operators: [ {
// CHECK: name: "main",
// CHECK: debug_metadata_index:
// CHECK: name: "cond_true",
// CHECK: debug_metadata_index:
// CHECK: name: "cond_false",
// CHECK: debug_metadata_index:
// CHECK: metadata: [ {
// CHECK: name: "debug_metadata",