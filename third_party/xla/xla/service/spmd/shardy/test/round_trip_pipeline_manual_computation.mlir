// RUN: sdy_opt %s -xla-sdy-round-trip-testing-pipeline -split-input-file 2>&1 | FileCheck %s

// Test ShardMap. We can assume a frontend framework like JAX will add the
// sdy shardings on the custom calls. Make sure when we round-trip we get the
// ManualComputationOp though.

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: sharding_hlo_string

// CHECK: sdy.mesh @mesh_1 = <"a"=4, "b"=2>
sdy.mesh @mesh_1 = <"a"=4, "b"=2>

// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<16x32xf32>) -> tensor<128x32xf32> {
  // CHECK:          %[[SHARD_MAP:.*]]:2 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_1, [{}, {}], replicated={"a", "b"}>] out_shardings=[<@mesh_1, [{"a", "b"}, {}]>, <@mesh_1, [{"b", "a"}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:       sdy.return %arg1, %arg1 : tensor<16x32xf32>, tensor<16x32xf32>
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> (tensor<128x32xf32>, tensor<128x32xf32>)
  // CHECK-NEXT:     %[[ADD:.*]] = mhlo.add %[[SHARD_MAP]]#0, %[[SHARD_MAP]]#1 : tensor<128x32xf32>
  // CHECK-NEXT:     return %[[ADD]] : tensor<128x32xf32>
  %0 = mhlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}], replicated={"a", "b"}>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = mhlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %2:2 = call @shmap_body_4(%1) : (tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>)
  %3 = mhlo.custom_call @Sharding(%2#0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %4 = mhlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a", "b"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<128x32xf32>
  %5 = mhlo.custom_call @Sharding(%2#1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %6 = mhlo.custom_call @SPMDShardToFullShape(%5) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"b", "a"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<128x32xf32>
  %7 = mhlo.add %4, %6 : tensor<128x32xf32>
  return %7 : tensor<128x32xf32>
}
// CHECK-NOT: func.func private @shmap_body_4
func.func private @shmap_body_4(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
  return %arg0, %arg0 : tensor<16x32xf32>, tensor<16x32xf32>
}
