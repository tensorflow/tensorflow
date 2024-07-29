// RUN: mlir-hlo-opt --stablehlo-ext-chlo-recompose-ops --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

// CHECK-LABEL: func @recompose_topk
func.func @recompose_topk(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // CHECK: %values, %indices = chlo.top_k(%arg0, k = 4) {largest = true} : tensor<5x16xf32> -> (tensor<?x?xf32>, tensor<?x?xi32>)
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = true}
  } : (tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @recompose_topk_invalid_attr
func.func @recompose_topk_invalid_attr(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // CHECK: stablehlo.custom_call @mhlo.topk
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = false}
  } : (tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @recompose_tan
func.func @recompose_tan(%arg0: tensor<16xf32>) -> tensor<?xf32> {
  // CHECK: %0 = chlo.tan %arg0 : tensor<16xf32> -> tensor<?xf32>
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "mhlo.tan",
    mhlo.attributes = {},
    mhlo.version = 1 : i64
  } : (tensor<16xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @recompose_erf
func.func @recompose_erf(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: %0 = chlo.erf %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  %0 = "stablehlo.custom_call"(%arg0) {
    backend_config = "",
    call_target_name = "mhlo.erf",
    mhlo.attributes = {},
    mhlo.version = 1 : i64
  } : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  func.return %0 : tensor<?x20x20xbf16>
}

