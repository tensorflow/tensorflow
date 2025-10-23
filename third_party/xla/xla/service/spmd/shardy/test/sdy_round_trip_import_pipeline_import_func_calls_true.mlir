// RUN: sdy_opt %s --split-input-file -xla-sdy-round-trip-import-pipeline='import-func-calls=true' 2>&1 | FileCheck %s

// CHECK-LABEL: func @non_flat_call_graph_all_inlineable
func.func @non_flat_call_graph_all_inlineable(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: %0 = sdy.named_computation<"foo">(%arg0)
  // CHECK: %1 = stablehlo.negate %0
  // CHECK: %2 = sdy.named_computation<"baz">(%1)
  // CHECK: return %2 : tensor<8xf32>
  %0 = call @foo(%arg0) {mhlo.frontend_attributes = {inlineable = "true"}} : (tensor<8xf32>) -> tensor<8xf32>
  %1 = stablehlo.negate %0 : tensor<8xf32>
  %2 = call @baz(%1) {mhlo.frontend_attributes = {inlineable = "true"}} : (tensor<8xf32>) -> tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8xf32>
  %1 = call @bar(%0) {mhlo.frontend_attributes = {inlineable = "true"}} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-NOT: func private @bar
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-NOT: func private @baz
func.func private @baz(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
// CHECK-LABEL: func @uninlineable_call
func.func @uninlineable_call(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: %0 = sdy.named_computation<"foo">(%arg0)
  // CHECK: return %0 : tensor<8xf32>
  %0 = call @foo(%arg0) {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
