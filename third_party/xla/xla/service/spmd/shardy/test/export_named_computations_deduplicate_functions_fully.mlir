// RUN: sdy_opt %s -xla-sdy-export-named-computations='dedup-functions-fully=true' -split-input-file | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_same_named_computations_different_shardings(
func.func @multiple_same_named_computations_different_shardings(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL0:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[CALL1:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %[[CALL1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CALL0]], %[[COPY]]
  // CHECK-NEXT: return %[[ADD]]
  %0 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {}]>] (%arg1: tensor<8x2xi32>) {
    %2 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
    sdy.return %2 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {"y"}]>] (%arg1: tensor<8x2xi32>) {
    %3 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %3 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %4 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %4 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-NEXT:  stablehlo.multiply %arg0, %arg0
// CHECK-SAME:  sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_same_named_computations_different_shardings_different_number_of_call_sites(
func.func @multiple_same_named_computations_different_shardings_different_number_of_call_sites(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL0:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %[[CALL0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[CALL1:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[CALL2:.*]] = call @baz(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: return %[[CALL2]] : tensor<8x2xi32>
  %0 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {}]>] (%arg1: tensor<8x2xi32>) {
    %2 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
    sdy.return %2 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {"y"}]>] (%arg1: tensor<8x2xi32>) {
    %3 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %3 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = sdy.named_computation<"baz">(%0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {"y"}]>] (%arg1: tensor<8x2xi32>) {
    %3 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %3 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
// CHECK-NEXT:  stablehlo.multiply %arg0, %arg0
// CHECK-SAME:  sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_same_named_computations_multiple_outputs_different_shardings(
func.func @multiple_same_named_computations_multiple_outputs_different_shardings(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL0:.*]]:2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %[[DIVIDE0:.*]] = stablehlo.divide %[[CALL0]]#0, %[[CALL0]]#1
  // CHECK-NEXT: %[[CALL1:.*]]:2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %[[COPY0:.*]] = mhlo.copy %[[CALL1]]#1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>}
  // CHECK-NEXT: %[[COPY1:.*]] = mhlo.copy %[[CALL1]]#0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[DIVIDE1:.*]] = stablehlo.divide %[[COPY1]], %[[COPY0]]
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[DIVIDE0]], %[[DIVIDE1]]
  // CHECK-NEXT: return %[[ADD]]
  %0:2 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>] (%arg1: tensor<8x2xi32>) {
    %5 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
    sdy.return %5, %5 : tensor<8x2xi32>, tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %1 = stablehlo.divide %0#0, %0#1 : tensor<8x2xi32>
  %2:2 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {"x"}]>] (%arg1: tensor<8x2xi32>) {
    %5 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %5, %5 : tensor<8x2xi32>, tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %3 = stablehlo.divide %2#0, %2#1 : tensor<8x2xi32>
  %4 = stablehlo.add %1, %3 : tensor<8x2xi32>
  return %4 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-NEXT:  stablehlo.multiply %arg0, %arg0
// CHECK-SAME:  sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @named_computations_same_funcs_two_same_manual_axes_different_shardings_one_without_manual_axes(
// CHECK-NEXT:  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %4 = func.call @foo(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %5 = mhlo.copy %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<4xf32>
// CHECK-NEXT:    sdy.return %5 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = call @foo_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %2 = sdy.manual_computation(%1) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %3 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %2 : tensor<8xf32>
func.func @named_computations_same_funcs_two_same_manual_axes_different_shardings_one_without_manual_axes(%arg0: tensor<8xf32>) -> tensor<8xf32> {
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
  %7 = sdy.manual_computation(%5) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg2: tensor<4xf32>) {
      %2 = stablehlo.abs %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
      sdy.return %2 : tensor<4xf32>
    } {xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %7 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {

// CHECK-LABEL: func private @foo_0(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
