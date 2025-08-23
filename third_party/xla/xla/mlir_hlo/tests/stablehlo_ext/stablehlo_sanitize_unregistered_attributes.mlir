// RUN: mlir-hlo-opt --stablehlo-ext-sanitize-unregistered-attributes --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: func @frontend_attr
// CHECK: mhlo.frontend_attributes = "Test"
func.func @frontend_attr(%arg0: tensor<2x2xi32> {jax.buffer_donor = true})
    -> tensor<2x2xi32> attributes {mhlo.frontend_attributes = "Test" : i64} {
  return %arg0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @unknown_attr
// CHECK-NOT: xla_tpu_user_reserved_hbm_bytes
func.func @unknown_attr(%arg0: tensor<2x2xi32> {jax.buffer_donor = true})
    -> tensor<2x2xi32> attributes {xla_tpu_user_reserved_hbm_bytes = 0 : i64} {
  return %arg0 : tensor<2x2xi32>
}
