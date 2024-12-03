// RUN: sdy_opt %s -xla-sdy-round-trip-export-pipeline -inline -xla-sdy-round-trip-testing-pipeline -split-input-file 2>&1 | FileCheck %s

// Test with a nested func op that gets inlined after first export.

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: xla.sdy.sharding

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @main(
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {}]>})
func.func @main(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {}]>}) {
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[ADD_0]], %[[ADD_0]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[MUL]], %[[MUL]] : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ADD_1]] : tensor<8x16xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  %1 = func.call @nested_func(%0) : (tensor<8x16xf32>) -> (tensor<8x16xf32>)
  %2 = stablehlo.add %1, %1 : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-NOT: func @nested_func
func.func @nested_func(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
