// RUN: xla-opt %s -split-input-file \
// RUN: -tensor-lower-to-triton \
// RUN: | FileCheck %s

//TODO(basioli): Consider fusing this and stablehlo_to_triton_lowering.mlir into xtile_to_triton_lowering.mlir

// CHECK: func @lower_bitcast(%[[ARG:.*]]: tensor<2x4x8xf32>) -> tensor<2x4x8xi32>
func.func @lower_bitcast(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xi32> {
  // CHECK: %[[RES:.*]] = tt.bitcast %[[ARG]] : tensor<2x4x8xf32> -> tensor<2x4x8xi32>
  %0 = tensor.bitcast %arg0 : tensor<2x4x8xf32> to tensor<2x4x8xi32>
  // CHECK: return %[[RES]] : tensor<2x4x8xi32>
  return %0 : tensor<2x4x8xi32>
}

// CHECK: func @lower_extract_on_one_element_tensor(%[[ARG:.*]]: tensor<1x1x1xf32>) -> f32
func.func @lower_extract_on_one_element_tensor(%arg0: tensor<1x1x1xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[RESHAPE:.*]] = tt.reshape %[[ARG]] allow_reorder : tensor<1x1x1xf32> -> tensor<1xf32>
  // CHECK: %[[RES:.*]] = "tt.reduce"(%[[RESHAPE]]) <{axis = 0 : i32}> ({
  // CHECK:   ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32):
  // CHECK:     %[[ADD:.*]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:     tt.reduce.return %[[ADD]] : f32
  // CHECK:   }) : (tensor<1xf32>) -> f32 
  %0 = tensor.extract %arg0[%c0, %c0, %c0] : tensor<1x1x1xf32>
  // CHECK: return %[[RES]] : f32
  return %0 : f32
}


// CHECK: func @lower_extract_on_multiple_element_tensor_falls_back_to_tensor(%[[ARG:.*]]: tensor<1x1x3xf32>) -> f32
func.func @lower_extract_on_multiple_element_tensor_falls_back_to_tensor(%arg0: tensor<1x1x3xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[RES:.*]] = tensor.extract %[[ARG]][%c0, %c0, %c0] : tensor<1x1x3xf32>
  %0 = tensor.extract %arg0[%c0, %c0, %c0] : tensor<1x1x3xf32>
  // CHECK: return %[[RES]] : f32
  return %0 : f32
}