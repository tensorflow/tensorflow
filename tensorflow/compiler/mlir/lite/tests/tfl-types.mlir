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

// Parser, printer and verifier tests for the TFLite element types
// (`!tfl.string`, `!tfl.quint8`, `!tfl.resource` and `!tfl.variant`), and for
// TFLite ops accepting both these types and the matching `tf_type` types.

// RUN: litert-opt -split-input-file -verify-diagnostics -tfl-runtime-verify %s | FileCheck %s
// RUN: litert-opt -split-input-file -verify-diagnostics -tfl-runtime-verify %s -mlir-print-op-generic | litert-opt -split-input-file | FileCheck %s --check-prefix=ROUNDTRIP

// CHECK-LABEL: printParse
// CHECK-SAME: tensor<2x!tfl.string>
// CHECK-SAME: tensor<2x!tfl.quint8>
// CHECK-SAME: tensor<!tfl.resource>
// CHECK-SAME: tensor<!tfl.resource<tensor<2xf32>>>
// CHECK-SAME: tensor<!tfl.resource<tensor<*xi32>, tensor<?x!tfl.string>>>
// CHECK-SAME: tensor<!tfl.variant>
// CHECK-SAME: tensor<!tfl.variant<tensor<3xi64>>>
// ROUNDTRIP-LABEL: printParse
// ROUNDTRIP-SAME: tensor<2x!tfl.string>
// ROUNDTRIP-SAME: tensor<2x!tfl.quint8>
// ROUNDTRIP-SAME: tensor<!tfl.resource>
// ROUNDTRIP-SAME: tensor<!tfl.resource<tensor<2xf32>>>
// ROUNDTRIP-SAME: tensor<!tfl.resource<tensor<*xi32>, tensor<?x!tfl.string>>>
// ROUNDTRIP-SAME: tensor<!tfl.variant>
// ROUNDTRIP-SAME: tensor<!tfl.variant<tensor<3xi64>>>
func.func @printParse(
    %arg0: tensor<2x!tfl.string>,
    %arg1: tensor<2x!tfl.quint8>,
    %arg2: tensor<!tfl.resource>,
    %arg3: tensor<!tfl.resource<tensor<2xf32>>>,
    %arg4: tensor<!tfl.resource<tensor<*xi32>, tensor<?x!tfl.string>>>,
    %arg5: tensor<!tfl.variant>,
    %arg6: tensor<!tfl.variant<tensor<3xi64>>>) {
  func.return
}

// -----

// expected-error @+1 {{expected a tensor subtype for !tfl.resource, got 'f32'}}
func.func @resourceNonTensorSubtype(%arg0: tensor<!tfl.resource<f32>>)

// -----

// expected-error @+1 {{expected a tensor subtype for !tfl.variant, got 'i32'}}
func.func @variantNonTensorSubtype(%arg0: tensor<!tfl.variant<tensor<2xf32>, i32>>)

// -----

// expected-error @+1 {{unknown TFL type: foo}}
func.func @unknownType(%arg0: tensor<!tfl.foo>)

// -----

//===----------------------------------------------------------------------===//
// TFLite ops with the TFLite types.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: gatherString
func.func @gatherString(%arg0 : tensor<3x!tfl.string>, %arg1 : tensor<2xi32>) -> tensor<2x!tfl.string> {
  // CHECK: "tfl.gather"(%arg0, %arg1) <{axis = 0 : i32}> : (tensor<3x!tfl.string>, tensor<2xi32>) -> tensor<2x!tfl.string>
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<3x!tfl.string>, tensor<2xi32>) -> tensor<2x!tfl.string>
  func.return %0 : tensor<2x!tfl.string>
}

// -----

func.func @gatherUnsupported2DString(%arg0 : tensor<?x?x!tfl.string>, %arg1 : tensor<?xi32>) -> tensor<*x!tfl.string> {
  // expected-error @+1 {{'tfl.gather' op expect 1d input when the given type is string, got 'tensor<?x?x!tfl.string>'}}
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32} : (tensor<?x?x!tfl.string>, tensor<?xi32>) -> tensor<*x!tfl.string>
  func.return %0 : tensor<*x!tfl.string>
}

// -----

// ui8 and quint8 map to the same TFLite type.
// CHECK-LABEL: stridedSliceQuint8
func.func @stridedSliceQuint8(%arg0: tensor<12x2x2x5xui8>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<1x2x2x5x!tfl.quint8> {
  // CHECK: "tfl.strided_slice"
  %0 = "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<12x2x2x5xui8>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5x!tfl.quint8>
  func.return %0 : tensor<1x2x2x5x!tfl.quint8>
}

// -----

// CHECK-LABEL: softmaxQuint8
func.func @softmaxQuint8(%arg0: tensor<2x!tfl.quint8>) -> tensor<2x!tfl.quint8> {
  // CHECK: "tfl.softmax"(%arg0) <{beta = 1.000000e+00 : f32}> : (tensor<2x!tfl.quint8>) -> tensor<2x!tfl.quint8>
  %0 = "tfl.softmax"(%arg0) {beta = 1.0 : f32} : (tensor<2x!tfl.quint8>) -> tensor<2x!tfl.quint8>
  func.return %0 : tensor<2x!tfl.quint8>
}

// -----

// CHECK-LABEL: variables
func.func @variables(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: "tfl.var_handle"() <{container = "", shared_name = "v"}> : () -> tensor<!tfl.resource<tensor<2xf32>>>
  %0 = "tfl.var_handle"() {container = "", shared_name = "v"} : () -> tensor<!tfl.resource<tensor<2xf32>>>
  // CHECK: "tfl.assign_variable"
  "tfl.assign_variable"(%0, %arg0) : (tensor<!tfl.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
  // CHECK: "tfl.read_variable"
  %1 = "tfl.read_variable"(%0) : (tensor<!tfl.resource<tensor<2xf32>>>) -> tensor<2xf32>
  func.return %1 : tensor<2xf32>
}

// -----

// CHECK-LABEL: hashtable
func.func @hashtable(%arg0: tensor<2x!tfl.string>, %arg1: tensor<2xi64>) -> tensor<2xi64> {
  // CHECK: "tfl.hashtable"() <{key_dtype = !tfl.string, table_id = 1 : i32, value_dtype = i64}> : () -> tensor<1x!tfl.resource>
  %0 = "tfl.hashtable"() {table_id = 1 : i32, key_dtype = !tfl.string, value_dtype = i64} : () -> tensor<1x!tfl.resource>
  // CHECK: "tfl.hashtable_import"
  "tfl.hashtable_import"(%0, %arg0, %arg1) : (tensor<1x!tfl.resource>, tensor<2x!tfl.string>, tensor<2xi64>) -> ()
  %cst = arith.constant dense<-1> : tensor<i64>
  // CHECK: "tfl.hashtable_find"
  %1 = "tfl.hashtable_find"(%0, %arg0, %cst) : (tensor<1x!tfl.resource>, tensor<2x!tfl.string>, tensor<i64>) -> tensor<2xi64>
  func.return %1 : tensor<2xi64>
}

// -----

func.func @readVariableNotResource(%arg0: tensor<!tfl.variant>) -> tensor<2xf32> {
  // expected-error @+1 {{'tfl.read_variable' op operand #0 must be tensor of resource values, but got 'tensor<!tfl.variant>'}}
  %0 = "tfl.read_variable"(%arg0) : (tensor<!tfl.variant>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// TFLite ops with the matching `tf_type` types.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: gatherTfString
func.func @gatherTfString(%arg0 : tensor<3x!tf_type.string>, %arg1 : tensor<2xi32>) -> tensor<2x!tf_type.string> {
  // CHECK: "tfl.gather"(%arg0, %arg1) <{axis = 0 : i32}> : (tensor<3x!tf_type.string>, tensor<2xi32>) -> tensor<2x!tf_type.string>
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<3x!tf_type.string>, tensor<2xi32>) -> tensor<2x!tf_type.string>
  func.return %0 : tensor<2x!tf_type.string>
}

// -----

// CHECK-LABEL: stridedSliceTfQuint8
func.func @stridedSliceTfQuint8(%arg0: tensor<12x2x2x5xui8>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<1x2x2x5x!tf_type.quint8> {
  // CHECK: "tfl.strided_slice"
  %0 = "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<12x2x2x5xui8>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5x!tf_type.quint8>
  func.return %0 : tensor<1x2x2x5x!tf_type.quint8>
}

// -----

// CHECK-LABEL: softmaxTfQuint8
func.func @softmaxTfQuint8(%arg0: tensor<2x!tf_type.quint8>) -> tensor<2x!tf_type.quint8> {
  // CHECK: "tfl.softmax"(%arg0) <{beta = 1.000000e+00 : f32}> : (tensor<2x!tf_type.quint8>) -> tensor<2x!tf_type.quint8>
  %0 = "tfl.softmax"(%arg0) {beta = 1.0 : f32} : (tensor<2x!tf_type.quint8>) -> tensor<2x!tf_type.quint8>
  func.return %0 : tensor<2x!tf_type.quint8>
}

// -----

// CHECK-LABEL: tfVariables
func.func @tfVariables(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: "tfl.var_handle"() <{container = "", shared_name = "v"}> : () -> tensor<!tf_type.resource<tensor<2xf32>>>
  %0 = "tfl.var_handle"() {container = "", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
  // CHECK: "tfl.assign_variable"
  "tfl.assign_variable"(%0, %arg0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
  // CHECK: "tfl.read_variable"
  %1 = "tfl.read_variable"(%0) : (tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<2xf32>
  func.return %1 : tensor<2xf32>
}

// -----

// Reference types are not supported on TFLite ops.
func.func @readVariableRef(%arg0: tensor<!tf_type.resourceref>) -> tensor<2xf32> {
  // expected-error @+1 {{'tfl.read_variable' op operand #0 must be tensor of resource values, but got 'tensor<!tf_type.resourceref>'}}
  %0 = "tfl.read_variable"(%arg0) : (tensor<!tf_type.resourceref>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Cast compatibility.
//===----------------------------------------------------------------------===//

// Resource types of the same kind are cast compatible regardless of subtypes.
// CHECK-LABEL: reshapeResourceSubtypes
func.func @reshapeResourceSubtypes(%arg0: tensor<1x!tfl.resource<tensor<2xf32>>>) -> tensor<1x!tfl.resource<tensor<3xi32>>> {
  %shape = arith.constant dense<1> : tensor<1xi32>
  // CHECK: "tfl.reshape"
  %0 = "tfl.reshape"(%arg0, %shape) : (tensor<1x!tfl.resource<tensor<2xf32>>>, tensor<1xi32>) -> tensor<1x!tfl.resource<tensor<3xi32>>>
  func.return %0 : tensor<1x!tfl.resource<tensor<3xi32>>>
}

// -----

// CHECK-LABEL: reshapeTfResourceSubtypes
func.func @reshapeTfResourceSubtypes(%arg0: tensor<1x!tf_type.resource<tensor<2xf32>>>) -> tensor<1x!tf_type.resource<tensor<3xi32>>> {
  %shape = arith.constant dense<1> : tensor<1xi32>
  // CHECK: "tfl.reshape"
  %0 = "tfl.reshape"(%arg0, %shape) : (tensor<1x!tf_type.resource<tensor<2xf32>>>, tensor<1xi32>) -> tensor<1x!tf_type.resource<tensor<3xi32>>>
  func.return %0 : tensor<1x!tf_type.resource<tensor<3xi32>>>
}

// -----

// CHECK-LABEL: reshapeVariantSubtypes
func.func @reshapeVariantSubtypes(%arg0: tensor<1x!tfl.variant<tensor<2xf32>>>) -> tensor<1x!tfl.variant> {
  %shape = arith.constant dense<1> : tensor<1xi32>
  // CHECK: "tfl.reshape"
  %0 = "tfl.reshape"(%arg0, %shape) : (tensor<1x!tfl.variant<tensor<2xf32>>>, tensor<1xi32>) -> tensor<1x!tfl.variant>
  func.return %0 : tensor<1x!tfl.variant>
}

// -----

// CHECK-LABEL: reshapeTfVariantSubtypes
func.func @reshapeTfVariantSubtypes(%arg0: tensor<1x!tf_type.variant<tensor<2xf32>>>) -> tensor<1x!tf_type.variant> {
  %shape = arith.constant dense<1> : tensor<1xi32>
  // CHECK: "tfl.reshape"
  %0 = "tfl.reshape"(%arg0, %shape) : (tensor<1x!tf_type.variant<tensor<2xf32>>>, tensor<1xi32>) -> tensor<1x!tf_type.variant>
  func.return %0 : tensor<1x!tf_type.variant>
}

// -----

// A TFLite type and the matching `tf_type` type are not cast compatible.
func.func @reshapeMixedResource(%arg0: tensor<1x!tfl.resource>) -> tensor<1x!tf_type.resource> {
  %shape = arith.constant dense<1> : tensor<1xi32>
  // expected-error @+1 {{'tfl.reshape' op requires 'output' type 'tensor<1x!tf_type.resource>' to be cast compatible with expected type 'tensor<1x!tfl.resource>'}}
  %0 = "tfl.reshape"(%arg0, %shape) : (tensor<1x!tfl.resource>, tensor<1xi32>) -> tensor<1x!tf_type.resource>
  func.return %0 : tensor<1x!tf_type.resource>
}

// -----

// Resource and variant types are not cast compatible.
func.func @reshapeResourceToVariant(%arg0: tensor<1x!tfl.resource>) -> tensor<1x!tfl.variant> {
  %shape = arith.constant dense<1> : tensor<1xi32>
  // expected-error @+1 {{'tfl.reshape' op requires 'output' type 'tensor<1x!tfl.variant>' to be cast compatible with expected type 'tensor<1x!tfl.resource>'}}
  %0 = "tfl.reshape"(%arg0, %shape) : (tensor<1x!tfl.resource>, tensor<1xi32>) -> tensor<1x!tfl.variant>
  func.return %0 : tensor<1x!tfl.variant>
}

// -----

// SameOperandsAndResultTypeResolveRef doesn't resolve reference types.
func.func @ceilRef(%arg0: tensor<2x!tf_type.f32ref>) -> tensor<2xf32> {
  // expected-error @+1 {{'tfl.ceil' op requires compatible element types for all operands and results}}
  %0 = "tfl.ceil"(%arg0) : (tensor<2x!tf_type.f32ref>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}
