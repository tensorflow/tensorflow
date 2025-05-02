// RUN: sdy_opt %s -xla-sdy-round-trip-testing-pipeline -split-input-file 2>&1 | FileCheck %s

// Test ShardMap. We can assume a frontend framework like JAX will add the
// sdy shardings on the custom calls. Make sure when we round-trip we get the
// ManualComputationOp though.

// ***************** Basic test *****************

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
  %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{}, {}], replicated={"a", "b"}>] out_shardings=[<@mesh_1, [{"a", "b"}, {}]>, <@mesh_1, [{"b", "a"}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<16x32xf32>) {
    sdy.return %arg1, %arg1 : tensor<16x32xf32>, tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> (tensor<128x32xf32>, tensor<128x32xf32>)
  %1 = stablehlo.add %0#0, %0#1 : tensor<128x32xf32>
  return %1 : tensor<128x32xf32>
}

// -----

// ***************** No inputs test *****************

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: sharding_hlo_string

// CHECK: sdy.mesh @mesh_1 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=4, "b"=2]>

// CHECK-LABEL: func.func @main
func.func @main() -> tensor<4xi64> {
  // CHECK:          %[[SHARD_MAP:.*]] = sdy.manual_computation()
  // CHECK-SAME{LITERAL}: in_shardings=[] out_shardings=[<@mesh_1, [{"b"}]>] manual_axes={"b"} () {
  // CHECK-NEXT:       %[[C:.*]] = sdy.constant dense<[2, 3]> : tensor<2xi64>
  // CHECK-NEXT:       sdy.return %[[C]] : tensor<2xi64>
  // CHECK-NEXT:     } : () -> tensor<4xi64>
  // CHECK-NEXT:     return %[[SHARD_MAP]] : tensor<4xi64>
  %0 = sdy.manual_computation() in_shardings=[] out_shardings=[<@mesh_1, [{"b"}]>] manual_axes={"b"} () {
    %1 = sdy.constant dense<[2, 3]> : tensor<2xi64>
    sdy.return %1 : tensor<2xi64>
  } : () -> tensor<4xi64>
  func.return %0 : tensor<4xi64>
}

// -----

// ***************** No outputs test *****************

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: sharding_hlo_string

// CHECK: sdy.mesh @mesh_1 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=4, "b"=2]>

// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<4xi64>) {
  // CHECK:          sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_1, [{"b"}]>] out_shardings=[] manual_axes={"b"} (%arg1: tensor<2xi64>) {
  // CHECK-NEXT:       stablehlo.custom_call @sdy_testonly(%arg1) {backend_config = "", has_side_effect = true, xla_shape = "()"} : (tensor<2xi64>) -> ()
  // CHECK-NEXT:       sdy.return
  // CHECK-NEXT:     } : (tensor<4xi64>) -> ()
  // CHECK-NEXT:     return
  sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{"b"}]>] out_shardings=[] manual_axes={"b"} (%arg1: tensor<2xi64>) {
    stablehlo.custom_call @sdy_testonly(%arg1) {has_side_effect = true} : (tensor<2xi64>) -> ()
    sdy.return
  } : (tensor<4xi64>) -> ()
  return
}


// -----

// ***************** No inputs no outputs test *****************

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: sharding_hlo_string

// CHECK-LABEL: func.func @main
func.func @main() {
  // CHECK:          sdy.manual_computation()
  // CHECK-SAME{LITERAL}: in_shardings=[] out_shardings=[] manual_axes={} () {
  // CHECK-NEXT:       sdy.return
  // CHECK-NEXT:     } : () -> ()
  // CHECK-NEXT:     return
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={} () {
    sdy.return
  } : () -> ()
  return
}
