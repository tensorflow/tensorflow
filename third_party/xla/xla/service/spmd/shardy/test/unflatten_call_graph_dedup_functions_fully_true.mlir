// RUN: sdy_opt %s -split-input-file -xla-sdy-unflatten-call-graph='dedup-functions-fully=true' | FileCheck %s

// CHECK-LABEL: func @singleton(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    return
func.func @singleton(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @ignore_operand_shardings(
// CHECK-SAME: %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>})
// CHECK-SAME: -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
func.func @ignore_operand_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[COPY]])
  // CHECK-SAME:   {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"},
  // CHECK-SAME:    random_attr = "random_value",
  // CHECK-SAME:    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-SAME:   : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[MOVE_TO_HOST:.*]] = stablehlo.custom_call @MoveToHost(%[[CALL]]) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", ?}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[MOVE_TO_HOST]] : tensor<8x2xi32>
  %0 = call @foo(%arg0) {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}, random_attr = "random_value", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.custom_call @MoveToHost(%0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", ?}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
// CHECK-LABEL: func @vanilla_call(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
func.func @vanilla_call(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = call @bar(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
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

func.func private @foo(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_same_shardings(
func.func @multiple_same_calls_same_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %1 : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_same_shardings_without_original_func_names(
func.func @multiple_same_calls_same_shardings_without_original_func_names(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %1 : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_0(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_different_shardings(
func.func @multiple_same_calls_different_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %2 = mhlo.copy %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: return %2 : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})

// CHECK-LABEL: func private @baz_0(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_same_shardings_calls_have_different_manual_computation_calls
func.func @multiple_same_calls_same_shardings_calls_have_different_manual_computation_calls(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %2 : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @baz_1(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}
// CHECK-LABEL: func @xla.sdy.inlinable_manual_computation_body(
func.func @xla.sdy.inlinable_manual_computation_body(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  return %arg0 : tensor<8x2xi32>
}
// CHECK:   func @xla.sdy.inlinable_manual_computation_body_0(
func.func @xla.sdy.inlinable_manual_computation_body_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  return %arg0 : tensor<8x2xi32>
}
// CHECK:   func @xla.sdy.inlinable_manual_computation_body_1(
func.func @xla.sdy.inlinable_manual_computation_body_1(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  return %arg0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = call @xla.sdy.inlinable_manual_computation_body(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = call @xla.sdy.inlinable_manual_computation_body_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @baz_1(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = call @xla.sdy.inlinable_manual_computation_body_1(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @non_flat_nested_calls_same_shardings
func.func @non_flat_nested_calls_same_shardings(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  // CHECK-NEXT: %2 = call @baz(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %3 = call @bar(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: return %3 : tensor<8xf32>
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %2 = call @baz(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %3 = call @bar(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
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
func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
func.func private @baz(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @bar_0(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
attributes { xla.sdy.original_func_name = "bar" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @non_flat_nested_calls_different_shardings
func.func @non_flat_nested_calls_different_shardings(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  // CHECK-NEXT: %2 = call @baz(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %3 = call @bar_0(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: return %3 : tensor<8xf32>
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %2 = call @baz(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %3 = call @bar_0(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
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
// CHECK-NEXT:    %1 = call @bar_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    %2 = mhlo.copy %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %2 : tensor<8xf32>
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

func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
func.func private @baz(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @bar_0(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>})
attributes { xla.sdy.original_func_name = "bar" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @non_flat_nested_calls_mixed_shardings
func.func @non_flat_nested_calls_mixed_shardings(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  // CHECK-NEXT: %2 = call @bar_0(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %3 = call @bar_0(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %4 = mhlo.copy %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  // CHECK-NEXT: return %4 : tensor<8xf32>
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %2 = call @bar_0(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %3 = call @bar_1(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
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
// CHECK-NEXT:    %1 = call @bar_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    return %1 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @bar_0(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @bar_1(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
func.func private @bar_0(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
attributes { xla.sdy.original_func_name = "bar" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @bar_1(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>})
attributes { xla.sdy.original_func_name = "bar" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_with_manual_axes_simple(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %1 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %1 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
func.func @calls_with_manual_axes_simple(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %1 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4xf32>) -> tensor<4xf32>
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
func.func private @foo(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_same_func_with_manual_axes(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %1 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %2 = func.call @foo(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %2 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
func.func @calls_same_func_with_manual_axes(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %1 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %2 = func.call @foo_0(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %2 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

func.func private @foo(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @foo_0(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_different_funcs_with_manual_axes_one_without(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %4 = func.call @foo(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %4 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = mhlo.copy %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  %2 = call @bar(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %2 : tensor<8xf32>
func.func @calls_different_funcs_with_manual_axes_one_without(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %3 = func.call @foo_0(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %3 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
func.func private @foo(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @foo_0(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
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
// CHECK-LABEL: func @calls_same_funcs_with_manual_axes_one_without(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %4 = func.call @foo(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %4 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = mhlo.copy %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  %2 = call @bar(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %2 : tensor<8xf32>
func.func @calls_same_funcs_with_manual_axes_one_without(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %3 = func.call @foo_0(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %3 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
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
func.func private @foo(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @foo_0(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_same_funcs_without_manual_axes_one_inside_manual_computation(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
// CHECK-NEXT:    %4 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    sdy.return %4 : tensor<8xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %4 = func.call @bar(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %4 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %2 = mhlo.copy %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  %3 = call @foo(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %3 : tensor<8xf32>
func.func @calls_same_funcs_without_manual_axes_one_inside_manual_computation(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %3 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %3 = func.call @bar(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %3 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %2 = call @foo_0(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %2 : tensor<8xf32>
}
// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @bar(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @foo_0(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_with_manual_axes(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %6 = func.call @bar(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %7 = func.call @bar(%6) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %7 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = mhlo.copy %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  %2 = call @foo(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %3 = call @foo(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %4 = call @foo(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %5 = mhlo.copy %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  return %5 : tensor<8xf32>
func.func @calls_with_manual_axes(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %4 = func.call @bar(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %5 = func.call @bar_0(%4) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %5 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = call @foo(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %2 = call @foo_0(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %3 = call @foo_1(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
}
// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @foo(
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

func.func private @bar(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @bar_0(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
attributes { xla.sdy.original_func_name = "bar" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @foo_0(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @foo_1(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_same_funcs_same_shardings_one_inside_manual_computation_without_manual_axes_and_one_outside(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
// CHECK-NEXT:    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    sdy.return %2 : tensor<8xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = call @foo(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %1 : tensor<8xf32>
func.func @calls_same_funcs_same_shardings_one_inside_manual_computation_without_manual_axes_and_one_outside(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %2 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = call @foo_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @foo_0(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @calls_same_funcs_same_shardings_inside_separate_manual_computation_without_manual_axes(
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
func.func @calls_same_funcs_same_shardings_inside_separate_manual_computation_without_manual_axes(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %2 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %2 = func.call @foo_0(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %2 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @foo_0(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_same_funcs_same_shardings_inside_separate_manual_computation_with_manual_axes(
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
func.func @calls_same_funcs_same_shardings_inside_separate_manual_computation_with_manual_axes(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %2 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %2 = func.call @foo_0(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %2 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

func.func private @foo(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @foo_0(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_same_funcs_same_shardings_one_nested_and_inside_manual_computation_without_manual_axes_and_one_outside(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
// CHECK-NEXT:    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    sdy.return %2 : tensor<8xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = call @bar_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %1 : tensor<8xf32>
func.func @calls_same_funcs_same_shardings_one_nested_and_inside_manual_computation_without_manual_axes_and_one_outside(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %2 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = call @bar_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
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
// CHECK-NEXT:    %0 = call @bar_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @bar_0(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = call @bar(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @bar_0(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>})
attributes { xla.sdy.original_func_name = "bar" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @nested_manual_computations(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK:       %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %4 = sdy.manual_computation(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={"y"} (%arg2: tensor<2xf32>) {
// CHECK-NEXT:      %7 = func.call @foo(%arg2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:        sdy.return %7 : tensor<2xf32>
// CHECK-NEXT:      } : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %5 = mhlo.copy %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : tensor<4xf32>
// CHECK-NEXT:      %6 = func.call @bar(%5) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      sdy.return %6 : tensor<4xf32>
// CHECK-NEXT:    } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x", "y"} (%arg1: tensor<2xf32>) {
// CHECK-NEXT:      %4 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      sdy.return %4 : tensor<2xf32>
// CHECK-NEXT:    } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    %2 = mhlo.copy %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    %3 = call @baz(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    return %3 : tensor<8xf32>
func.func @nested_manual_computations(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %3 = sdy.manual_computation(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={"y"} (%arg2: tensor<2xf32>) {
      %5 = func.call @foo(%arg2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>} : (tensor<2xf32>) -> tensor<2xf32>
      sdy.return %5 : tensor<2xf32>
    } : (tensor<4xf32>) -> tensor<4xf32>
    %4 = func.call @bar(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %4 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x", "y"} (%arg1: tensor<2xf32>) {
    %3 = func.call @foo_0(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>} : (tensor<2xf32>) -> tensor<2xf32>
    sdy.return %3 : tensor<2xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %2 = call @baz(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %2 : tensor<8xf32>
}
// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>}
// CHECK-SAME:      -> (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<2xf32>
// CHECK-NEXT:    return %0 : tensor<2xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @baz(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:    return %0 : tensor<8xf32>
// CHECK-NEXT:  }

func.func private @foo(%arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>}) -> (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<2xf32>
  return %0 : tensor<2xf32>
}
func.func private @bar(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @foo_0(%arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>}) -> (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x", "y"}>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<2xf32>
  return %0 : tensor<2xf32>
}
func.func private @baz(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_same_funcs_on_the_same_manual_axes_different_shardings(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK-NEXT:  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %4 = func.call @foo(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %5 = mhlo.copy %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : tensor<4xf32>
// CHECK-NEXT:    sdy.return %5 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = mhlo.copy %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  %2 = call @bar(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %2 : tensor<8xf32>
func.func @calls_same_funcs_on_the_same_manual_axes_different_shardings(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %2 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %3 = func.call @foo_0(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %3 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {

// CHECK-LABEL: func private @foo_0(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
func.func private @foo(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @foo_0(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_different_shardings
func.func @multiple_same_calls_different_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL0:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[CALL1:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %[[CALL1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CALL0]], %[[COPY]]
  // CHECK-NEXT: return %[[ADD]]
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-NEXT:  stablehlo.multiply %arg0, %arg0
// CHECK-SAME:  sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>}

// CHECK-LABEL: func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:    return %0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_different_shardings_different_number_of_call_sites
func.func @multiple_same_calls_different_shardings_different_number_of_call_sites(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL0:.*]] = call @baz_1(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[COPY0:.*]] = mhlo.copy %[[CALL0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @baz_1(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[CALL2:.*]] = call @baz_1(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: return %[[CALL2]] : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @baz_1(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_0(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @baz_1(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_different_shardings_different_number_of_call_sites_one_called_twice
func.func @multiple_same_calls_different_shardings_different_number_of_call_sites_one_called_twice(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[CALL0:.*]] = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[COPY0:.*]] = mhlo.copy %[[CALL0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[CALL2:.*]] = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: return %[[CALL2]] : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_0(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_multiple_outputs_different_shardings
func.func @multiple_same_calls_multiple_outputs_different_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL0:.*]]:2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %[[DIVIDE0:.*]] = stablehlo.divide %[[CALL0]]#0, %[[CALL0]]#1
  // CHECK-NEXT: %[[CALL1:.*]]:2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  // CHECK-NEXT: %[[COPY0:.*]] = mhlo.copy %[[CALL1]]#1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[COPY1:.*]] = mhlo.copy %[[CALL1]]#0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[DIVIDE1:.*]] = stablehlo.divide %[[COPY1]], %[[COPY0]]
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[DIVIDE0]], %[[DIVIDE1]]
  // CHECK-NEXT: return %[[ADD]]
  %0:2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %1 = stablehlo.divide %0#0, %0#1 : tensor<8x2xi32>
  %2:2 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %3 = stablehlo.divide %2#0, %2#1 : tensor<8x2xi32>
  %4 = stablehlo.add %1, %3 : tensor<8x2xi32>
  return %4 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})

// CHECK-LABEL: func private @baz_0(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>})

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0, %0 : tensor<8x2xi32>, tensor<8x2xi32>
}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0, %0 : tensor<8x2xi32>, tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_same_funcs_two_same_manual_axes_different_shardings_one_without_manual_axes(
// CHECK-NEXT:  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %4 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %5 = func.call @foo(%4) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %6 = mhlo.copy %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : tensor<4xf32>
// CHECK-NEXT:    sdy.return %6 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = mhlo.copy %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  %2 = call @bar(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %3 = sdy.manual_computation(%2) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %4 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %4 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %3 : tensor<8xf32>
func.func @calls_same_funcs_two_same_manual_axes_different_shardings_one_without_manual_axes(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %4 = func.call @foo_0(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %4 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %2 = sdy.manual_computation(%1) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %3 = func.call @foo_1(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %3 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {

// CHECK-LABEL: func private @foo_0(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
func.func private @foo(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @foo_0(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @foo_1(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_different_shardings_without_original_func_names
func.func @multiple_same_calls_different_shardings_without_original_func_names(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL0:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[CALL1:.*]] = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CALL0]], %[[CALL1]]
  // CHECK-NEXT: return %[[ADD]]
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-NEXT:  stablehlo.multiply %arg0, %arg0
// CHECK-SAME:  sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>}

// CHECK-LABEL: func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:    return %0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_different_shardings_different_number_of_call_sites_without_original_func_names
func.func @multiple_same_calls_different_shardings_different_number_of_call_sites_without_original_func_names(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL0:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[CALL1:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[COPY0:.*]] = mhlo.copy %[[CALL1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[COPY1:.*]] = mhlo.copy %[[CALL0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL2:.*]] = call @baz(%[[COPY1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[COPY2:.*]] = mhlo.copy %[[CALL2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: return %[[COPY2]] : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @baz(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_0(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_1(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func private @baz_1(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_multiple_outputs_different_shardings_without_original_func_names
func.func @multiple_same_calls_multiple_outputs_different_shardings_without_original_func_names(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL0:.*]]:2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %[[DIVIDE0:.*]] = stablehlo.divide %[[CALL0]]#0, %[[CALL0]]#1
  // CHECK-NEXT: %[[CALL1:.*]]:2 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  // CHECK-NEXT: %[[DIVIDE1:.*]] = stablehlo.divide %[[CALL1]]#0, %[[CALL1]]#1
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[DIVIDE0]], %[[DIVIDE1]]
  // CHECK-NEXT: return %[[ADD]]
  %0:2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %1 = stablehlo.divide %0#0, %0#1 : tensor<8x2xi32>
  %2:2 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %3 = stablehlo.divide %2#0, %2#1 : tensor<8x2xi32>
  %4 = stablehlo.add %1, %3 : tensor<8x2xi32>
  return %4 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})

// CHECK-LABEL: func private @baz_0(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>})

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0, %0 : tensor<8x2xi32>, tensor<8x2xi32>
}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0, %0 : tensor<8x2xi32>, tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_same_funcs_two_same_manual_axes_different_shardings_one_without_manual_axes_without_original_func_names(
// CHECK-NEXT:  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %4 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %5 = mhlo.copy %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : tensor<4xf32>
// CHECK-NEXT:    %6 = func.call @foo_0(%5) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %6 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = mhlo.copy %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  %2 = call @bar(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %3 = sdy.manual_computation(%2) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %4 = func.call @foo_1(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %4 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %3 : tensor<8xf32>
func.func @calls_same_funcs_two_same_manual_axes_different_shardings_one_without_manual_axes_without_original_func_names(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    %4 = func.call @foo_0(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %4 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %2 = sdy.manual_computation(%1) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %3 = func.call @foo_1(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %3 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {

// CHECK-LABEL: func private @foo_0(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {

// CHECK-LABEL: func private @foo_1(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}
// CHECK-SAME:      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
func.func private @foo(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @foo_0(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @foo_1(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_same_argument_and_func_input_sharding(
func.func @single_call_same_argument_and_func_input_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_different_argument_and_func_input_sharding(
func.func @single_call_different_argument_and_func_input_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @baz(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_no_func_input_sharding(
func.func @single_call_no_func_input_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @baz(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_different_argument_and_func_input_sharding_argument_reused(
func.func @single_call_different_argument_and_func_input_sharding_argument_reused(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @baz(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CALL]], %arg0 : tensor<8x2xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.add %0, %arg0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_different_shardings_different_number_of_call_sites_multiple_func_origins
func.func @multiple_same_calls_different_shardings_different_number_of_call_sites_multiple_func_origins(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[CALL0:.*]] = call @baz_1(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[COPY0:.*]] = mhlo.copy %[[CALL0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @baz_1(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[CALL2:.*]] = call @baz_1(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[CALL3:.*]] = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[CALL4:.*]] = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[CALL5:.*]] = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[CALL6:.*]] = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: return %[[CALL6]] : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @baz_1(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %3 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %4 = call @foo_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %5 = call @foo_1(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %6 = call @foo_2(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %6 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_0(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_1(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func private @baz_1(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @foo(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func private @foo(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @foo_0(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func private @foo_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @foo_1(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func private @foo_1(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @foo_2(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func private @foo_2(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
attributes { xla.sdy.original_func_name = "foo" } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_with_manual_axes(
func.func @single_call_with_manual_axes(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// CHECK-LABEL: func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
// CHECK-SAME:     -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
attributes { xla.sdy.manual_axes = #sdy<manual_axes{"x"}> } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @same_func_different_manual_axes(
func.func @same_func_different_manual_axes(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL0:.*]] = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"y"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL0]] : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"y"}>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
// CHECK-SAME:     -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) {
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"x"}>})
attributes { xla.sdy.manual_axes = #sdy<manual_axes{"x"}> } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"y"}>})
// CHECK-SAME:     -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"y"}>}) {
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"y"}>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"y"}>})
attributes { xla.sdy.manual_axes = #sdy<manual_axes{"y"}> } {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}


// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_func_result_empty_sharding_call_has_sharding(
func.func @single_call_func_result_empty_sharding_call_has_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %1 = mhlo.copy %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: return %1
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_func_result_empty_sharding_call_has_sharding(
func.func @single_call_func_result_empty_sharding_call_has_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %1 = mhlo.copy %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: return %1
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @two_calls_same_origin_one_call_with_empty_sharding(
func.func @two_calls_same_origin_one_call_with_empty_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %2 = mhlo.copy %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %3 = stablehlo.add %0, %2 : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32>
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @three_calls_same_origin_func_with_two_calls_results_no_sharding(
func.func @three_calls_same_origin_func_with_two_calls_results_no_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %1 = mhlo.copy %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %2 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %3 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %4 = stablehlo.add %1, %2 : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %3 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %3 : tensor<8x2xi32>
}

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32>
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @three_calls_same_origin_func_with_one_call_results_no_sharding(
func.func @three_calls_same_origin_func_with_one_call_results_no_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %1 = mhlo.copy %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %2 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %3 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %4 = stablehlo.add %1, %2 : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %3 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %3 : tensor<8x2xi32>
}

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_both_call_and_func_without_result_sharding(
func.func @single_call_both_call_and_func_without_result_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @baz(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %0
  %0 = call @baz(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @three_calls_same_origin_func_with_one_call_results_no_out_sharding(
func.func @three_calls_same_origin_func_with_one_call_results_no_out_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %3 = mhlo.copy %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %4 = stablehlo.add %0, %1 : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @baz_0(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %3 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %3 : tensor<8x2xi32>
}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32>
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @three_calls_same_origin_func_with_one_call_results_no_out_sharding(
func.func @three_calls_same_origin_func_with_one_call_results_no_out_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %3 = mhlo.copy %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %4 = stablehlo.add %0, %1 : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @baz_0(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %3 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %3 : tensor<8x2xi32>
}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32>
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
//CHECK-LABEL: func @three_calls_same_origin_func_with_two_calls_results_no_out_sharding(
func.func @three_calls_same_origin_func_with_two_calls_results_no_out_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %1 = mhlo.copy %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %2 = call @baz_0(%arg0) :
  // CHECK-NEXT: %3 = call @baz_0(%arg0) :
  // CHECK-NEXT: %4 = stablehlo.add %1, %2 : tensor<8x2xi32>
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @baz_0(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %3 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %3 : tensor<8x2xi32>
}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32>
attributes { xla.sdy.original_func_name = "baz" } {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
