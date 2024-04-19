// RUN: mlir-hlo-opt --shape-legalize-to-hlo=legalize-constraints=true -reconcile-unrealized-casts -canonicalize --split-input-file --verify-diagnostics %s | FileCheck %s
// This test verifies e2e lowering of cstr ops result is correct for constant inputs.

// -----

// CHECK-LABEL: func.func @mhlo_cstr_reshapable_true
func.func @mhlo_cstr_reshapable_true(%arg0: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %0 = arith.constant 16 : index
  %1 = mhlo.constant dense<[-1, 4, 2]> : tensor<3xi32>
  %2 = mhlo.cstr_reshapable %0, %1 : (index, tensor<3xi32>) -> !shape.witness
  %3 = shape.assuming %2 -> tensor<?x2xf32> {
    shape.assuming_yield %arg0 : tensor<?x2xf32>
  }
  func.return %3 : tensor<?x2xf32>
  //      CHECK: %[[TRUE:.*]] = mhlo.constant dense<true> : tensor<i1>
  // CHECK-NEXT: mhlo.custom_call @shape_assertion(%[[TRUE]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: func.func @mhlo_cstr_reshapable_has_residual
func.func @mhlo_cstr_reshapable_has_residual(%arg0: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %0 = arith.constant 19 : index
  %1 = mhlo.constant dense<[-1, 4]> : tensor<2xi32>
  %2 = mhlo.cstr_reshapable %0, %1 : (index, tensor<2xi32>) -> !shape.witness
  %3 = shape.assuming %2 -> tensor<?x2xf32> {
    shape.assuming_yield %arg0 : tensor<?x2xf32>
  }
  func.return %3 : tensor<?x2xf32>
  //      CHECK: %[[FALSE:.*]] = mhlo.constant dense<false> : tensor<i1>
  // CHECK-NEXT: mhlo.custom_call @shape_assertion(%[[FALSE]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: func.func @mhlo_cstr_reshapable_2_dynamic_dims
func.func @mhlo_cstr_reshapable_2_dynamic_dims(%arg0: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %0 = arith.constant 20 : index
  %1 = mhlo.constant dense<[-1, 4, -1]> : tensor<3xi32>
  %2 = mhlo.cstr_reshapable %0, %1 : (index, tensor<3xi32>) -> !shape.witness
  %3 = shape.assuming %2 -> tensor<?x2xf32> {
    shape.assuming_yield %arg0 : tensor<?x2xf32>
  }
  func.return %3 : tensor<?x2xf32>
  //      CHECK: %[[FALSE:.*]] = mhlo.constant dense<false> : tensor<i1>
  // CHECK-NEXT: mhlo.custom_call @shape_assertion(%[[FALSE]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: func.func @mhlo_cstr_reshapable_static_true
func.func @mhlo_cstr_reshapable_static_true(%arg0: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %0 = arith.constant 20 : index
  %1 = mhlo.constant dense<[1, 4, 5]> : tensor<3xi32>
  %2 = mhlo.cstr_reshapable %0, %1 : (index, tensor<3xi32>) -> !shape.witness
  %3 = shape.assuming %2 -> tensor<?x2xf32> {
    shape.assuming_yield %arg0 : tensor<?x2xf32>
  }
  func.return %3 : tensor<?x2xf32>
  //      CHECK: %[[TRUE:.*]] = mhlo.constant dense<true> : tensor<i1>
  // CHECK-NEXT: mhlo.custom_call @shape_assertion(%[[TRUE]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: func.func @mhlo_cstr_reshapable_static_false
func.func @mhlo_cstr_reshapable_static_false(%arg0: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %0 = arith.constant 21 : index
  %1 = mhlo.constant dense<[1, 4, 5]> : tensor<3xi32>
  %2 = mhlo.cstr_reshapable %0, %1 : (index, tensor<3xi32>) -> !shape.witness
  %3 = shape.assuming %2 -> tensor<?x2xf32> {
    shape.assuming_yield %arg0 : tensor<?x2xf32>
  }
  func.return %3 : tensor<?x2xf32>
  //      CHECK: %[[FALSE:.*]] = mhlo.constant dense<false> : tensor<i1>
  // CHECK-NEXT: mhlo.custom_call @shape_assertion(%[[FALSE]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<?x2xf32>
}
