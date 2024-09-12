// RUN: sdy_opt %s -xla-sdy-round-trip-testing-pipeline -split-input-file 2>&1 | FileCheck %s

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// COMPILER API OP TESTS.
// These would be needed to work for round-tripping in JAX integration.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Basic test with func arg sharding

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: xla.sdy.sharding

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @main
func.func @main(
  // CHECK: %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}p4]>})
  %arg0: tensor<8x16xf32>           {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}p4]>}
  ) -> (tensor<8x16xf32>) {
  %0 = mhlo.add %arg0, %arg0 : tensor<8x16xf32>
  %1 = mhlo.add %0, %0 : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// -----

// Test that a sharding on the result of a function is kept around. Due to how
// MHLO->HLO conversion works discarding any frontend attributes on the function
// results, we copy the sharding to a temporary custom call before discarding it
// after the round-trip.

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: xla.sdy.sharding

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @main
func.func @main(
  // CHECK: %arg0: tensor<8x16xf32>)
  %arg0: tensor<8x16xf32>
  // CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}p4]>}) {
  ) -> (tensor<8x16xf32>              {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}p4]>}) {
  // CHECK: mhlo.add %arg0, %arg0 : tensor<8x16xf32>
  %0 = mhlo.add %arg0, %arg0 : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

// Test conflict between op and result sharding. We keep both around.

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: xla.sdy.sharding

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @main
func.func @main(
  // CHECK: %arg0: tensor<8x16xf32>)
  %arg0: tensor<8x16xf32>
  // CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}p4]>}) {
  ) -> (tensor<8x16xf32>              {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}p4]>}) {
  // CHECK:  sdy.sharding_constraint %arg0 <@mesh, [{"b", ?}, {"a"}p4]> : tensor<8x16xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"b", ?}, {"a"}p4]> : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

// Test that a result sharding whose value is the function argument. Due to how
// MHLO->HLO conversion works discarding any frontend attributes on the function
// results, we copy the sharding to a temporary custom call before discarding it
// after the round-trip.

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: xla.sdy.sharding

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @main
func.func @main(
  // CHECK: %arg0: tensor<8x16xf32>)
  %arg0: tensor<8x16xf32>
  // CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}p4]>}) {
  ) -> (tensor<8x16xf32>              {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}p4]>}) {
  return %arg0 : tensor<8x16xf32>
}

// -----

// Test sharding just on the ReturnOp operand's defining op but not FuncOp
// result.

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: xla.sdy.sharding

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL:      @main(
// CHECK-SAME:   %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}p4]>},
// CHECK-SAME:   %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>
// CHECK-SAME:   ) -> tensor<8x8xf32> {
func.func @main(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}p4]>},
  %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK:      %[[ADD:.*]] = mhlo.add %arg0, %arg1 : tensor<8x8xf32>
  // CHECK-NEXT: %[[WSC:.*]] = sdy.sharding_constraint %0 <@mesh, [{}, {"c", ?}p1]> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[WSC]] : tensor<8x8xf32>
  %0 = mhlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"c", ?}p1]> : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// Test ShardingConstraintOp.

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: sharding_hlo_string

// CHECK: sdy.mesh @mesh = <["data"=2]>
sdy.mesh @mesh = <["data"=2]>

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK:  sdy.sharding_constraint %arg0 <@mesh, [{"data", ?}, {?}]> :  tensor<8x8xf32>
  %0 = sdy.sharding_constraint %arg0       <@mesh, [{"data", ?}, {?}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// NON COMPILER API OP TESTS.
// These would be needed to work if we wanted to round-trip after propagation,
// run GSPMD propagation a little, and come back.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Test FuncOp with multiple result shardings.

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: xla.sdy.sharding

// CHECK: sdy.mesh @mesh_2 = <["x"=8, "y"=4]>
sdy.mesh @mesh_2 = <["x"=8, "y"=4]>

// CHECK-LABEL: func @main
func.func @main(
  // CHECK: %arg0: tensor<8x16xf32>) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", ?}, {"y"}p4]>}, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{?}, {"y"}p4]>}, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {"y"}p1]>}) {
  %arg0: tensor<8x16xf32>) ->           (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", ?}, {"y"}p4]>}, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{?}, {"y"}p4]>}, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {"y"}p1]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = mhlo.add %arg0, %arg0 : tensor<8x16xf32>
  %0 = mhlo.add %arg0, %arg0 : tensor<8x16xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = mhlo.custom_call @sdy_testonly(%arg0) {backend_config = "", xla_shape = "(f32[8,16]{1,0}, f32[8,16]{1,0})"} : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
  %1:2 = mhlo.custom_call @sdy_testonly(%arg0) : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
  // CHECK-NEXT: return %[[ADD]], %[[CUSTOM_CALL]]#0, %[[CUSTOM_CALL]]#1
  return %0, %1#0, %1#1 : tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>
}

// -----

// CHECK: sdy.mesh @mesh = <["x"=2]>
sdy.mesh @mesh = <["x"=2]>

// Test WhileOp with lifted free variables and sinked constants.

// CHECK-LABEL: func @main
func.func @main(
    %arg0: tensor<32x96xf32>,
    %arg1: tensor<32x96xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}})
    -> tensor<32x96xf32> {
  // CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0>
  // CHECK-NEXT: %[[C32:.*]] = sdy.constant dense<32>
  // CHECK-NEXT: %[[SC:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{?}, {?}]>
  // CHECK-NEXT: %[[WHILE:.*]]:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]])
  // CHECK-NEXT:   cond {
  // CHECK-NEXT:   %[[COND:.*]] = mhlo.compare LT, %iterArg_0, %[[C32]]
  // CHECK-NEXT:   mhlo.return %[[COND]]
  // CHECK-NEXT: } do {
  // CHECK-DAG:    %[[C1:.*]] = sdy.constant dense<1>
  // CHECK-DAG:    %[[ADD_0:.*]] = mhlo.add %iterArg_0, %[[C1]]
  // CHECK-DAG:    %[[ADD_1:.*]] = mhlo.add %iterArg, %[[SC]]
  // CHECK-NEXT:   mhlo.return %[[ADD_1]], %[[ADD_0]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[WHILE]]#0
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<32> : tensor<i32>
  %2:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %3 = mhlo.compare LT, %iterArg_0, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    mhlo.return %3 : tensor<i1>
  } do {
    %3 = sdy.constant dense<1> : tensor<i32>
    %4 = mhlo.add %iterArg_0, %3 : tensor<i32>
    %5 = mhlo.add %iterArg, %arg1 : tensor<32x96xf32>
    mhlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %2#0 : tensor<32x96xf32>
}

// TODO(b/335481977): Add more tests for MHLO ops. So far tested all SDY
// compiler APIs other than shard as/like (doesn't exist yet). See
// round_trip_pipeline_manual_computation.mlir for ManualComputationOp tests.
