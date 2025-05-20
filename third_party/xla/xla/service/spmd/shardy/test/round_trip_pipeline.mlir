// RUN: sdy_opt %s -xla-sdy-round-trip-testing-pipeline -split-input-file 2>&1 | FileCheck %s

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// COMPILER API OP TESTS.
// These would be needed to work for round-tripping in JAX integration.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Basic test with no meshes or shardings

// CHECK-NOT: sdy.mesh

// CHECK-LABEL: func @main(
// CHECK-SAME: %arg0: tensor<8x16xf32>)
func.func @main(
  %arg0: tensor<8x16xf32>) -> (tensor<8x16xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// -----

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
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// -----

// Test that a sharding on the result of a function is kept around. Due to how
// StableHLO->HLO conversion works discarding any frontend attributes on the function
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
  // CHECK: stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
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
// StableHLO->HLO conversion works discarding any frontend attributes on the function
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
  // CHECK:      %[[ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  // CHECK-NEXT: %[[WSC:.*]] = sdy.sharding_constraint %0 <@mesh, [{}, {"c", ?}p1]> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[WSC]] : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
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
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @sdy_testonly(%arg0) {backend_config = "", xla_shape = "(f32[8,16]{1,0}, f32[8,16]{1,0})"} : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
  %1:2 = stablehlo.custom_call @sdy_testonly(%arg0) : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
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
    %arg1: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> tensor<32x96xf32> {
  // CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0>
  // CHECK-NEXT: %[[SC:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{?}, {?}]>
  // CHECK-NEXT: %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]])
  // CHECK-NEXT:   cond {
  // CHECK-NEXT:   %[[C32:.*]] = sdy.constant dense<32>
  // CHECK-NEXT:   %[[COND:.*]] = stablehlo.compare LT, %iterArg_0, %[[C32]]
  // CHECK-NEXT:   stablehlo.return %[[COND]]
  // CHECK-NEXT: } do {
  // CHECK-DAG:    %[[C1:.*]] = sdy.constant dense<1>
  // CHECK-DAG:    %[[ADD_0:.*]] = stablehlo.add %iterArg_0, %[[C1]]
  // CHECK-DAG:    %[[ADD_1:.*]] = stablehlo.add %iterArg, %[[SC]]
  // CHECK-NEXT:   stablehlo.return %[[ADD_1]], %[[ADD_0]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[WHILE]]#0
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<32> : tensor<i32>
  %2:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %3 = stablehlo.compare LT, %iterArg_0, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %3 : tensor<i1>
  } do {
    %3 = sdy.constant dense<1> : tensor<i32>
    %4 = stablehlo.add %iterArg_0, %3 : tensor<i32>
    %5 = stablehlo.add %iterArg, %arg1 : tensor<32x96xf32>
    stablehlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %2#0 : tensor<32x96xf32>
}

// -----

// Test that sharding group op is preserved under import and export passes.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8x16xf32>) -> (tensor<8x16xf32>) {
  // CHECK: sdy.sharding_group %arg0 group_id=13 : tensor<8x16xf32>
  sdy.sharding_group %arg0 group_id=13 : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8x16xf32>) -> (tensor<8x16xf32>) {
  // CHECK: sdy.propagation_barrier %arg0 allowed_direction=BACKWARD : tensor<8x16xf32>
  %r = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD : tensor<8x16xf32>
  return %r : tensor<8x16xf32>
}

// -----

// Test call with backend config and multiple results. This is what JAX would
// emit in the frontend, and then we'd convert it to a NamedComputationOp when
// coming back.

func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK:      %[[NC:.*]]:2 = sdy.named_computation<"g.2.{{[0-9]}}">(%arg0) (%arg1: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MUL:.*]] = stablehlo.multiply %arg1, %arg1 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MUL]], %[[MUL]] : tensor<8x2xi32>, tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  // CHECK-NEXT: %[[HOST:.*]] = stablehlo.custom_call @MoveToHost(%[[NC]]#0) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[HOST]] : tensor<8x2xi32>
  %0:2 = call @g.2(%arg0) {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}, mhlo.sharding = "{{maximal device=0}, {replicated}}"} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %1 = stablehlo.custom_call @MoveToHost(%0#0) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: g.2
func.func private @g.2(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>) {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<8x2xi32>
  return %0, %0 : tensor<8x2xi32>, tensor<8x2xi32>
}

// TODO(b/335481977): Add more tests for StableHLO ops. So far tested all SDY
// compiler APIs other than shard as/like (doesn't exist yet). See
// round_trip_pipeline_manual_computation.mlir for ManualComputationOp tests.
