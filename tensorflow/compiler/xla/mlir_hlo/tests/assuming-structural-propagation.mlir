// RUN: mlir-hlo-opt -hlo-legalize-to-lhlo %s | FileCheck %s

// Test that assuming ops propagate tensor types.
// CHECK-LABEL: func @shape_assuming_tensor
func.func @shape_assuming_tensor(%arg0: tensor<?xf16>) -> tensor<?xf16> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
  %1 = shape.const_witness true
  // CHECK: shape.assuming %{{.*}} -> (memref<?xf16>)
  %2 = shape.assuming %1 -> (tensor<?xf16>) {
    // CHECK: "lmhlo.maximum"(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?xf16>, memref<?xf16>, memref<?xf16>) -> ()
    %7 = mhlo.maximum %arg0, %arg0 : tensor<?xf16>
    // CHECK: shape.assuming_yield %{{.*}} : memref<?xf16>
    shape.assuming_yield %7 : tensor<?xf16>
  }
  func.return %2 : tensor<?xf16>
}
