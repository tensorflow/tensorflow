// RUN: sdy_opt %s -xla-sdy-round-trip-testing-pipeline -split-input-file 2>&1 | FileCheck %s

// Test ShardMap. We can assume a frontend framework like JAX will add the
// sdy shardings on the custom calls. Make sure when we round-trip we get the
// ManualComputationOp though.

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: sharding_hlo_string

// CHECK: sdy.mesh @mesh_1 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=4, "b"=2]>

// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<16x32xf32>) -> tensor<128x32xf32> {
  // CHECK:          %[[SHARD_MAP:.*]]:2 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_1, [{}, {}], replicated={"a", "b"}>] out_shardings=[<@mesh_1, [{"a", "b"}, {}]>, <@mesh_1, [{"b", "a"}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:       sdy.return %arg1, %arg1 : tensor<16x32xf32>, tensor<16x32xf32>
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> (tensor<128x32xf32>, tensor<128x32xf32>)
  // CHECK-NEXT:     %[[ADD:.*]] = stablehlo.add %[[SHARD_MAP]]#0, %[[SHARD_MAP]]#1 : tensor<128x32xf32>
  // CHECK-NEXT:     return %[[ADD]] : tensor<128x32xf32>
  %0 = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1:2 = call @local_xla.sdy.manual_computation_body(%0) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {}], replicated={\\\22a\\\22, \\\22b\\\22}>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22, \\\22b\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\\\22a\\\22, \\\22b\\\22}, {}]>, <@mesh_1, [{\\\22b\\\22, \\\22a\\\22}, {}]>]>"}} : (tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>)
  %2:2 = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%1#0, %1#1) : (tensor<16x32xf32>, tensor<16x32xf32>) -> (tensor<128x32xf32>, tensor<128x32xf32>)
  %3 = stablehlo.add %2#0, %2#1 : tensor<128x32xf32>
  return %3 : tensor<128x32xf32>
}
// CHECK-NOT: func.func private @local_xla.sdy.manual_computation_body
func.func private @local_xla.sdy.manual_computation_body(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
  return %arg0, %arg0 : tensor<16x32xf32>, tensor<16x32xf32>
}
