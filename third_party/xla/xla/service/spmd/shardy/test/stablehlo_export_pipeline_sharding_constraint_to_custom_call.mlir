// RUN: sdy_opt %s -xla-sdy-export-ops='keep-hlo-sharding-constraints=true' 2>&1 | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @sharding_constraint_to_sharding_custom_call
func.func @sharding_constraint_to_sharding_custom_call(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: %0 = stablehlo.custom_call @Sharding(%arg0)
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {?}]>]>}
  // CHECK-SAME: (tensor<8x8xf32>) -> tensor<8x8xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
