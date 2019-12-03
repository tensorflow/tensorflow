// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: ENTRY
func @main(%arg0: tensor<3x4xi32>, %arg1: tensor<4x5xi32>) -> tensor<3x5xi32> {
  // Simple einsum is lowered to HLO dot op.
  // CHECK: dot(s32[3,4] %{{.*}}, s32[4,5] %{{.*}}), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %0 = "xla_hlo.einsum"(%arg0, %arg1) {einsum_config = "ab,bc->ac"} : (tensor<3x4xi32>, tensor<4x5xi32>) -> tensor<3x5xi32>
  return %0 : tensor<3x5xi32>
}
