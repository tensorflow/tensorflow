// RUN: mlir-hlo-opt %s -stablehlo-ext-refine-shapes --split-input-file 2>&1 | FileCheck %s

// Only the operand is manual.

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @main
// CHECK-SAME:  (%arg0: tensor<16x32xf32>) -> tensor<8x32xf32>
func.func @main(%arg0: tensor<16x32xf32>) -> tensor<?x32xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%0)
  // CHECK-SAME:     in_shardings=[<@mesh, [{"a", ?}, {?}]>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{?}, {?}], replicated={"a"}>]
  // CHECK-SAME:     manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
  // CHECK-NEXT:  %[[ADD_2:.*]] = stablehlo.add %arg1, %arg1 : tensor<8x32xf32>
  // CHECK-NEXT:  sdy.return %[[ADD_2]] : tensor<8x32xf32>
  // CHECK-NEXT: } : (tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: return %[[MC]] : tensor<8x32xf32>
  %0 = stablehlo.add %arg0, %arg0 : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<?x32xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"a", ?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}], replicated={"a"}>] manual_axes={"a"} (%arg1: tensor<?x32xf32>) {
    %2 = stablehlo.add %arg1, %arg1 : tensor<?x32xf32>
    sdy.return %2 : tensor<?x32xf32>
  } : (tensor<?x32xf32>) -> tensor<?x32xf32>
  return %1: tensor<?x32xf32>
}

// -----

// Only the result is manual.

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @main
// CHECK-SAME:  (%arg0: tensor<16x32xf32>) -> tensor<32x32xf32>
func.func @main(%arg0: tensor<16x32xf32>) -> tensor<?x32xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%0)
  // CHECK-SAME:     in_shardings=[<@mesh, [{?}, {?}], replicated={"a"}>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{"a", ?}, {?}]>]
  // CHECK-SAME:     manual_axes={"a"} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:  %[[ADD_2:.*]] = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
  // CHECK-NEXT:  sdy.return %[[ADD_2]] : tensor<16x32xf32>
  // CHECK-NEXT: } : (tensor<16x32xf32>) -> tensor<32x32xf32>
  // CHECK-NEXT: return %[[MC]] : tensor<32x32xf32>
  %0 = stablehlo.add %arg0, %arg0 : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<?x32xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{?}, {?}], replicated={"a"}>] out_shardings=[<@mesh, [{"a", ?}, {?}]>] manual_axes={"a"} (%arg1: tensor<?x32xf32>) {
    %2 = stablehlo.add %arg1, %arg1 : tensor<?x32xf32>
    sdy.return %2 : tensor<?x32xf32>
  } : (tensor<?x32xf32>) -> tensor<?x32xf32>
  return %1: tensor<?x32xf32>
}

// -----

// Both operand and result are manual.

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @main
// CHECK-SAME:  (%arg0: tensor<16x32xf32>) -> tensor<16x32xf32>
func.func @main(%arg0: tensor<16x32xf32>) -> tensor<?x32xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%0)
  // CHECK-SAME:     in_shardings=[<@mesh, [{"a", ?}, {?}]>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{"a", ?}, {?}]>]
  // CHECK-SAME:     manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
  // CHECK-NEXT:  %[[ADD_2:.*]] = stablehlo.add %arg1, %arg1 : tensor<8x32xf32>
  // CHECK-NEXT:  sdy.return %[[ADD_2]] : tensor<8x32xf32>
  // CHECK-NEXT: } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT: return %[[MC]] : tensor<16x32xf32>
  %0 = stablehlo.add %arg0, %arg0 : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<?x32xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"a", ?}, {?}]>] out_shardings=[<@mesh, [{"a", ?}, {?}]>] manual_axes={"a"} (%arg1: tensor<?x32xf32>) {
    %2 = stablehlo.add %arg1, %arg1 : tensor<?x32xf32>
    sdy.return %2 : tensor<?x32xf32>
  } : (tensor<?x32xf32>) -> tensor<?x32xf32>
  return %1: tensor<?x32xf32>
}

// -----

// The dimension being refined is not the one which is manually sharded.

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @main
// CHECK-SAME:  (%arg0: tensor<4x4xf32>) -> tensor<4x4xf32>
func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x?xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 : tensor<4x4xf32>
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ABS]])
  // CHECK-SAME:     in_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:     manual_axes={"x"} (%arg1: tensor<2x4xf32>) {
  // CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<2x4xf32>
  // CHECK-NEXT:  sdy.return %[[ADD]] : tensor<2x4xf32>
  // CHECK-NEXT: } : (tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: return %[[MC]] : tensor<4x4xf32>
  %9 = stablehlo.abs %arg0 : (tensor<4x4xf32>) -> tensor<4x?xf32>
  %0 = sdy.manual_computation(%9) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x?xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<2x?xf32>
    sdy.return %1 : tensor<2x?xf32>
  } : (tensor<4x?xf32>) -> tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}

// -----

// Body of named computation has all SDY operations.

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @main
// CHECK-SAME:  (%arg0: tensor<16x32xf32>) -> tensor<16x32xf32>
func.func @main(%arg0: tensor<16x32xf32>) -> tensor<?x32xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 : tensor<16x32xf32>
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%[[ABS]]) (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:   %[[SC:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{"b"}, {?}]> : tensor<16x32xf32>
  // CHECK-NEXT:   %[[RESHARD:.*]] = sdy.reshard %[[SC]] <@mesh, [{"b", "a"}, {?}]> : tensor<16x32xf32>
  // CHECK-NEXT:   sdy.sharding_group %[[RESHARD]] group_id=0 : tensor<16x32xf32>
  // CHECK-NEXT:   sdy.return %[[RESHARD]] : tensor<16x32xf32>
  // CHECK-NEXT: } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT: return %[[NC]] : tensor<16x32xf32>
  %0 = stablehlo.abs %arg0 : (tensor<16x32xf32>) -> tensor<?x32xf32>
  %1 = sdy.named_computation<"foo">(%0) (%arg1: tensor<?x32xf32>) {
    %2 = sdy.sharding_constraint %arg1 <@mesh, [{"b"}, {?}]> : tensor<?x32xf32>
    %3 = sdy.reshard %2 <@mesh, [{"b", "a"}, {?}]> : tensor<?x32xf32>
    sdy.sharding_group %3 group_id=0 : tensor<?x32xf32>
    sdy.return %3 : tensor<?x32xf32>
  } : (tensor<?x32xf32>) -> tensor<?x32xf32>
  return %1: tensor<?x32xf32>
}

// -----

// Body of manual computation has all SDY operations.

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @main
// CHECK-SAME:  (%arg0: tensor<16x32xf32>) -> tensor<16x32xf32>
func.func @main(%arg0: tensor<16x32xf32>) -> tensor<?x32xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 : tensor<16x32xf32>
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%0)
  // CHECK-SAME:     in_shardings=[<@mesh, [{?}, {?}]>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{?}, {?}]>]
  // CHECK-SAME:     manual_axes={} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:   %[[SC:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{"b"}, {?}]> : tensor<16x32xf32>
  // CHECK-NEXT:   %[[RESHARD:.*]] = sdy.reshard %[[SC]] <@mesh, [{"b", "a"}, {?}]> : tensor<16x32xf32>
  // CHECK-NEXT:   sdy.sharding_group %[[RESHARD]] group_id=0 : tensor<16x32xf32>
  // CHECK-NEXT:   sdy.return %[[RESHARD]] : tensor<16x32xf32>
  // CHECK-NEXT: } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT: return %[[MC]] : tensor<16x32xf32>
  %0 = stablehlo.abs %arg0 : (tensor<16x32xf32>) -> tensor<?x32xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg1: tensor<?x32xf32>) {
    %2 = sdy.sharding_constraint %arg1 <@mesh, [{"b"}, {?}]> : tensor<?x32xf32>
    %3 = sdy.reshard %2 <@mesh, [{"b", "a"}, {?}]> : tensor<?x32xf32>
    sdy.sharding_group %3 group_id=0 : tensor<?x32xf32>
    sdy.return %3 : tensor<?x32xf32>
  } : (tensor<?x32xf32>) -> tensor<?x32xf32>
  return %1: tensor<?x32xf32>
}

// -----

// Body of the manual computation has a call.
// TODO(b/385323320): the function is not being fully refined due to the call.

sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func @main
// CHECK-SAME:  (%arg0: tensor<4xf32>) -> tensor<?xf32>
func.func @main(%arg0: tensor<4xf32>) -> (tensor<?xf32>) {
  // CHECK-NEXT: %[[C:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK-NEXT: %[[CONVERT:.*]] = stablehlo.bitcast_convert %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[CONVERT]], %[[C]])
  // CHECK-SAME:     in_shardings=[<@mesh, [{"a"}]>, <@mesh, []>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{?}], replicated={"a"}>]
  // CHECK-SAME:     manual_axes={"a"} (%arg1: tensor<2xf32>, %arg2: tensor<i32>) {
  // CHECK-NEXT:   %[[CALL:.*]] = func.call @refine_call_callee(%arg2, %arg1) : (tensor<i32>, tensor<2xf32>) -> tensor<?xf32>
  // CHECK-NEXT:   sdy.return %[[CALL:.*]] : tensor<?xf32>
  // CHECK-NEXT: } : (tensor<4xf32>, tensor<i32>) -> tensor<?xf32>
  // CHECK-NEXT: return %[[MC]] : tensor<?xf32>
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<4xf32>) -> tensor<?xf32>
  %1 = stablehlo.constant dense<4> : tensor<i32>
  %2 = sdy.manual_computation(%0, %1) in_shardings=[<@mesh, [{"a"}]>, <@mesh, []>] out_shardings=[<@mesh, [{?}], replicated={"a"}>] manual_axes={"a"} (%arg1: tensor<?xf32>, %arg2: tensor<i32>) {
    %3 = func.call @refine_call_callee(%arg2, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
    sdy.return %3 : tensor<?xf32>
  } : (tensor<?xf32>, tensor<i32>) -> tensor<?xf32>
  return %2: tensor<?xf32>
}

// CHECK-LABEL: func @refine_call_callee
// CHECK-SAME:  (%arg0: tensor<i32>, %arg1: tensor<2xf32>) -> tensor<?xf32>
func.func @refine_call_callee(%arg0: tensor<i32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[IOTA:.*]] = stablehlo.dynamic_iota %[[RESHAPE]], dim = 0 : (tensor<1xi32>) -> tensor<?xf32>
  // CHECK-NEXT: return %[[IOTA]] : tensor<?xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}


// -----

// Body of the named computation has a call.
// TODO(b/385323320): the function is not being fully refined due to the call.

sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func @main
// CHECK-SAME:  (%arg0: tensor<4xf32>) -> tensor<?xf32>
func.func @main(%arg0: tensor<4xf32>) -> (tensor<?xf32>) {
  // CHECK-NEXT: %[[C:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK-NEXT: %[[CONVERT:.*]] = stablehlo.bitcast_convert %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%[[CONVERT]], %[[C]]) (%arg1: tensor<4xf32>, %arg2: tensor<i32>) {
  // CHECK-NEXT:   %[[CALL:.*]] = func.call @refine_call_callee(%arg2, %arg1) : (tensor<i32>, tensor<4xf32>) -> tensor<?xf32>
  // CHECK-NEXT:   sdy.return %[[CALL:.*]] : tensor<?xf32>
  // CHECK-NEXT: } : (tensor<4xf32>, tensor<i32>) -> tensor<?xf32>
  // CHECK-NEXT: return %[[NC]] : tensor<?xf32>
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<4xf32>) -> tensor<?xf32>
  %1 = stablehlo.constant dense<4> : tensor<i32>
  %2 = sdy.named_computation<"foo">(%0, %1) (%arg1: tensor<?xf32>, %arg2: tensor<i32>) {
    %3 = func.call @refine_call_callee(%arg2, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
    sdy.return %3 : tensor<?xf32>
  } : (tensor<?xf32>, tensor<i32>) -> tensor<?xf32>
  return %2: tensor<?xf32>
}

// CHECK-LABEL: func @refine_call_callee
// CHECK-SAME:  (%arg0: tensor<i32>, %arg1: tensor<4xf32>) -> tensor<?xf32>
func.func @refine_call_callee(%arg0: tensor<i32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[IOTA:.*]] = stablehlo.dynamic_iota %[[RESHAPE]], dim = 0 : (tensor<1xi32>) -> tensor<?xf32>
  // CHECK-NEXT: return %[[IOTA]] : tensor<?xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}
