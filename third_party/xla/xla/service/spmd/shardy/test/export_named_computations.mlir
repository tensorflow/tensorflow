// RUN: sdy_opt %s -xla-sdy-export-named-computations -split-input-file | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

// Note we don't override the block argument shardings of the function
// @ignore_operand_shardings, but we set the argument shardings on the call
// to @foo.
// CHECK-LABEL: func @ignore_operand_shardings(
// CHECK-SAME: %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>})
// CHECK-SAME: -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
func.func @ignore_operand_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%arg0)
  // CHECK-SAME:   {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"},
  // CHECK-SAME:    random_attr = "random_value",
  // CHECK-SAME:    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-SAME:   : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[MOVE_TO_HOST:.*]] = stablehlo.custom_call @MoveToHost(%[[CALL]]) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", ?}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[MOVE_TO_HOST]] : tensor<8x2xi32>
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {}]>] (%arg1: tensor<8x2xi32>) {
    %2 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %2 : tensor<8x2xi32>
  } {random_attr = "random_value", mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.custom_call @MoveToHost(%0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", ?}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func @vanilla_named_computation(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
func.func @vanilla_named_computation(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<8x2xi32>) {
    %1 = stablehlo.multiply %arg1, %arg1 : tensor<8x2xi32>
    sdy.return %1 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:    (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
// CHECK-NEXT:    %[[MULT:.*]] = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:    return %0 : tensor<8x2xi32>

// CHECK-LABEL: func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
// CHECK-NEXT:    %[[MULT:.*]] = stablehlo.multiply %arg0, %arg0 : tensor<8x2xi32>
// CHECK-NEXT:    return %0 : tensor<8x2xi32>

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_same_named_computations_same_shardings(
func.func @multiple_same_named_computations_same_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %1 : tensor<8x2xi32>
  %0 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {}]>] (%arg1: tensor<8x2xi32>) {
    %2 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %2 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {}]>] (%arg1: tensor<8x2xi32>) {
    %3 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %3 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_same_named_computations_different_shardings(
func.func @multiple_same_named_computations_different_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %1 : tensor<8x2xi32>
  %0 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {}]>] (%arg1: tensor<8x2xi32>) {
    %2 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %2 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {"y"}]>] (%arg1: tensor<8x2xi32>) {
    %3 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %3 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})

// CHECK-LABEL: func private @baz_0(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_same_named_computations_same_shardings_named_computations_have_different_manual_computation_calls(
func.func @multiple_same_named_computations_same_shardings_named_computations_have_different_manual_computation_calls(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %2 : tensor<8x2xi32>
  %0 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] (%arg1: tensor<8x2xi32>) {
    %3 = func.call @xla.sdy.inlinable_manual_computation_body(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    sdy.return %3 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] (%arg1: tensor<8x2xi32>) {
    %3 = func.call @xla.sdy.inlinable_manual_computation_body_0(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    sdy.return %3 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] (%arg1: tensor<8x2xi32>) {
    %3 = func.call @xla.sdy.inlinable_manual_computation_body_1(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    sdy.return %3 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-LABEL: func @xla.sdy.inlinable_manual_computation_body(
func.func @xla.sdy.inlinable_manual_computation_body(
  %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
 -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  return %arg0 : tensor<8x2xi32>
}

// CHECK-NOT:   func @xla.sdy.inlinable_manual_computation_body_0(
func.func @xla.sdy.inlinable_manual_computation_body_0(
  %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
 -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  return %arg0 : tensor<8x2xi32>
}

// CHECK-NOT:   func @xla.sdy.inlinable_manual_computation_body_1(
func.func @xla.sdy.inlinable_manual_computation_body_1(
  %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
 -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  return %arg0 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @non_flat_nested_named_computations_same_shardings(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
func.func @non_flat_nested_named_computations_same_shardings(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  // CHECK-NEXT: %2 = call @baz(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %3 = call @bar(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: return %3 : tensor<8xf32>
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg1: tensor<8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
    %2 = sdy.named_computation<"bar">(%1) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg2: tensor<8xf32>) {
      %3 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
      sdy.return %3 : tensor<8xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %2 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %4 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %5 = sdy.named_computation<"baz">(%4) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg1: tensor<8xf32>) {
    %6 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
    sdy.return %6 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %7 = sdy.named_computation<"bar">(%5) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg1: tensor<8xf32>) {
    %8 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
    sdy.return %8 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %7 : tensor<8xf32>
}

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:   %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:   %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:   return %1 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @baz(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @non_flat_nested_named_computations_different_shardings(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
func.func @non_flat_nested_named_computations_different_shardings(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  // CHECK-NEXT: %2 = call @baz(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %3 = call @bar_0(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: return %3 : tensor<8xf32>
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg1: tensor<8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
    %2 = sdy.named_computation<"bar">(%1) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg2: tensor<8xf32>) {
      %3 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
      sdy.return %3 : tensor<8xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %2 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %4 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %5 = sdy.named_computation<"baz">(%4) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg1: tensor<8xf32>) {
    %6 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
    sdy.return %6 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %7 = sdy.named_computation<"bar">(%5) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<8xf32>) {
    %8 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
    sdy.return %8 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %7 : tensor<8xf32>
}

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    return %1 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @baz(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @bar_0(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @non_flat_nested_named_computations_mixed_shardings(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
func.func @non_flat_nested_named_computations_mixed_shardings(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  // CHECK-NEXT: %2 = call @bar(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %3 = call @bar_0(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: return %3 : tensor<8xf32>
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg1: tensor<8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
    %2 = sdy.named_computation<"bar">(%1) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg2: tensor<8xf32>) {
      %3 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
      sdy.return %3 : tensor<8xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %2 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %4 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %5 = sdy.named_computation<"bar">(%4) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg1: tensor<8xf32>) {
    %6 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
    sdy.return %6 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %7 = sdy.named_computation<"bar">(%5) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<8xf32>) {
    %8 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
    sdy.return %8 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %7 : tensor<8xf32>
}

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    return %1 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @bar_0(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_with_manual_axes_simple(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %1 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %1 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
func.func @named_computations_with_manual_axes_simple(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %2 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %2 : tensor<4xf32>
    } : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %1 : tensor<4xf32>
  } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_same_func_with_manual_axes(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %1 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %2 = func.call @foo(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %2 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
func.func @named_computations_same_func_with_manual_axes(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %2 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %2 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %3 = sdy.named_computation<"foo">(%1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %4 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %4 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %3 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }


// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_different_funcs_with_manual_axes_one_without(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %3 = func.call @foo(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %3 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %1 : tensor<8xf32>
func.func @named_computations_different_funcs_with_manual_axes_one_without(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %2 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %2 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %3 = sdy.named_computation<"foo">(%1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %4 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %4 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %3 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %5 = sdy.named_computation<"bar">(%0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<8xf32>) {
    %6 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
    sdy.return %6 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %5 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_same_funcs_with_manual_axes_one_without(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %3 = func.call @foo(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %3 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = call @foo_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %1 : tensor<8xf32>
func.func @named_computations_same_funcs_with_manual_axes_one_without(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %2 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %2 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %3 = sdy.named_computation<"foo">(%1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %4 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %4 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %3 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %5 = sdy.named_computation<"foo">(%0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<8xf32>) {
    %6 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
    sdy.return %6 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %5 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @foo_0(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_same_funcs_without_manual_axes_one_inside_manual_computation(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
// CHECK-NEXT:    %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    sdy.return %3 : tensor<8xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %3 = func.call @foo_0(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %3 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %2 = call @foo(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %2 : tensor<8xf32>
func.func @named_computations_same_funcs_without_manual_axes_one_inside_manual_computation(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<8xf32>) {
      %2 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<8xf32>
      sdy.return %2 : tensor<8xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %3 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %4 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %5 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %5 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %4 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %6 = sdy.named_computation<"foo">(%3) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<8xf32>) {
    %7 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
    sdy.return %7 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %6 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @foo_0(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_with_manual_axes(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %4 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %5 = func.call @foo(%4) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %5 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = call @foo_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %2 = call @foo_0(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %3 = call @foo_1(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %3 : tensor<8xf32>
func.func @named_computations_with_manual_axes(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %2 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %2 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %3 = sdy.named_computation<"foo">(%1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %4 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %4 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %3 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %5 = sdy.named_computation<"foo">(%0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<8xf32>) {
    %6 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
    sdy.return %6 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %7 = sdy.named_computation<"foo">(%5) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<8xf32>) {
    %8 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
    sdy.return %8 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %9 = sdy.named_computation<"foo">(%7) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] (%arg1: tensor<8xf32>) {
    %10 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : tensor<8xf32>
    sdy.return %10 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %9 : tensor<8xf32>
}


// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @foo_0(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @foo_1(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_same_funcs_same_shardings_one_inside_manual_computation_without_manual_axes_and_one_outside(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
// CHECK-NEXT:    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    sdy.return %2 : tensor<8xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = call @foo(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %1 : tensor<8xf32>
func.func @named_computations_same_funcs_same_shardings_one_inside_manual_computation_without_manual_axes_and_one_outside(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<8xf32>) {
      %2 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
      sdy.return %2 : tensor<8xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %3 = sdy.named_computation<"foo">(%0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<8xf32>) {
    %4 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
    sdy.return %4 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_same_funcs_same_shardings_inside_separate_manual_computation_without_manual_axes(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
// CHECK-NEXT:    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    sdy.return %2 : tensor<8xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
// CHECK-NEXT:    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    sdy.return %2 : tensor<8xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %1 : tensor<8xf32>
func.func @named_computations_same_funcs_same_shardings_inside_separate_manual_computation_without_manual_axes(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<8xf32>) {
      %2 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
      sdy.return %2 : tensor<8xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %3 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<8xf32>) {
      %2 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
      sdy.return %2 : tensor<8xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_same_funcs_same_shardings_inside_separate_manual_computation_with_manual_axes(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %2 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %2 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %1 : tensor<8xf32>
func.func @named_computations_same_funcs_same_shardings_inside_separate_manual_computation_with_manual_axes(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %4 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %5 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %5 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %4 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %3 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %4 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %5 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %5 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %4 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }


// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_same_funcs_same_shardings_one_nested_and_inside_manual_computation_without_manual_axes_and_one_outside(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
// CHECK-NEXT:    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    sdy.return %2 : tensor<8xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %1 : tensor<8xf32>
func.func @named_computations_same_funcs_same_shardings_one_nested_and_inside_manual_computation_without_manual_axes_and_one_outside(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<8xf32>) {
      %2 = sdy.named_computation<"bar">(%arg2) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg3: tensor<8xf32>) {
        %3 = stablehlo.abs %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
        sdy.return %3 : tensor<8xf32>
      } : (tensor<8xf32>) -> tensor<8xf32>
      sdy.return %2 : tensor<8xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %4 = sdy.named_computation<"bar">(%0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<8xf32>) {
    %5 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
    sdy.return %5 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %4: tensor<8xf32>
}

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = call @bar(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @nested_manual_computations(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %3 = sdy.manual_computation(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={"y"} (%arg2: tensor<2xf32>) {
// CHECK-NEXT:      %5 = func.call @foo(%arg2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:        sdy.return %5 : tensor<2xf32>
// CHECK-NEXT:      } : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %4 = func.call @foo_0(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      sdy.return %4 : tensor<4xf32>
// CHECK-NEXT:    } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x", "y"} (%arg1: tensor<2xf32>) {
// CHECK-NEXT:      %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      sdy.return %3 : tensor<2xf32>
// CHECK-NEXT:    } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    %2 = call @foo_1(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    return %2 : tensor<8xf32>
func.func @nested_manual_computations(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={"y"} (%arg2: tensor<2xf32>) {
      %2 = sdy.named_computation<"foo">(%arg2) in_shardings=[<@mesh, [{}]>] out_shardings=[<@mesh, [{}]>] (%arg3: tensor<2xf32>) {
        %3 = stablehlo.abs %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<2xf32>
        sdy.return %3 : tensor<2xf32>
      } {xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>} : (tensor<2xf32>) -> tensor<2xf32>
      sdy.return %2 : tensor<2xf32>
    } : (tensor<4xf32>) -> tensor<4xf32>
    %4 = sdy.named_computation<"foo">(%1) in_shardings=[<@mesh, [{}]>] out_shardings=[<@mesh, [{}]>] (%arg2: tensor<4xf32>) {
      %5 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<4xf32>
      sdy.return %5 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %4 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %6 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x", "y"} (%arg1: tensor<2xf32>) {
    %7 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{}]>] out_shardings=[<@mesh, [{}]>] (%arg2: tensor<2xf32>) {
      %8 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<2xf32>
      sdy.return %8 : tensor<2xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>} : (tensor<2xf32>) -> tensor<2xf32>
    sdy.return %7 : tensor<2xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %9 = sdy.named_computation<"foo">(%6) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg1: tensor<8xf32>) {
    %10 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
    sdy.return %10 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %9 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>}
// CHECK-SAME:      -> (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<2xf32>
// CHECK-NEXT:    return %0 : tensor<2xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @foo_0(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @foo_1(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_same_funcs_two_same_manual_axes_different_shardings_one_without_manual_axes(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %3 = func.call @foo_0(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %3 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = call @foo_1(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %1 : tensor<8xf32>
func.func @named_computations_same_funcs_two_same_manual_axes_different_shardings_one_without_manual_axes(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %2 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %2 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %3 = sdy.named_computation<"foo">(%1) in_shardings=[<@mesh, [{}]>] out_shardings=[<@mesh, [{}]>] (%arg2: tensor<4xf32>) {
      %4 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %4 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %3 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %5 = sdy.named_computation<"foo">(%0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<8xf32>) {
    %6 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
    sdy.return %6 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %5 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {

// CHECK-LABEL: func private @foo_0(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {

// CHECK-LABEL: func private @foo_1(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {

