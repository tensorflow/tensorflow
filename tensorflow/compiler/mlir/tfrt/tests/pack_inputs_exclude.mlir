// RUN: tf-tfrt-opt %s -pack-inputs="slices=1,0,24,2,28,16" | FileCheck %s

// Tests advanced slice packing with gaps/padding and unmerged variables:
// - Input 0 is preserved.
// - Inputs 1 and 2 are packed into a single i8 buffer with a 4-byte gap (start 0 and start 28).
// - Inputs 3 and 4 are skipped and remain unchanged.

module {
  // CHECK-LABEL: func.func @main
  // CHECK-SAME: (%arg0: tensor<10x10xf32>, %arg1: tensor<5x2xf32>, %arg2: tensor<6x1xf32>, %arg3: tensor<44xi8>) -> (tensor<10x10xf32>, tensor<2x3xf32>, tensor<4x1xf32>, tensor<5x2xf32>, tensor<6x1xf32>)
  func.func @main(%arg0: tensor<10x10xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<4x1xf32>, %arg3: tensor<5x2xf32>, %arg4: tensor<6x1xf32>) -> (tensor<10x10xf32>, tensor<2x3xf32>, tensor<4x1xf32>, tensor<5x2xf32>, tensor<6x1xf32>) attributes {ifrt.function} {
    // CHECK: %[[SLICE0:.*]] = stablehlo.slice %arg3 [0:24] : (tensor<44xi8>) -> tensor<24xi8>
    // CHECK: %[[RESHAPE0:.*]] = stablehlo.reshape %[[SLICE0]] : (tensor<24xi8>) -> tensor<2x3x4xi8>
    // CHECK: %[[BITCAST0:.*]] = stablehlo.bitcast_convert %[[RESHAPE0]] : (tensor<2x3x4xi8>) -> tensor<2x3xf32>
    // CHECK: %[[SLICE1:.*]] = stablehlo.slice %arg3 [28:44] : (tensor<44xi8>) -> tensor<16xi8>
    // CHECK: %[[RESHAPE1:.*]] = stablehlo.reshape %[[SLICE1]] : (tensor<16xi8>) -> tensor<4x1x4xi8>
    // CHECK: %[[BITCAST1:.*]] = stablehlo.bitcast_convert %[[RESHAPE1]] : (tensor<4x1x4xi8>) -> tensor<4x1xf32>
    // CHECK: return %arg0, %[[BITCAST0]], %[[BITCAST1]], %arg1, %arg2 : tensor<10x10xf32>, tensor<2x3xf32>, tensor<4x1xf32>, tensor<5x2xf32>, tensor<6x1xf32>
    return %arg0, %arg1, %arg2, %arg3, %arg4 : tensor<10x10xf32>, tensor<2x3xf32>, tensor<4x1xf32>, tensor<5x2xf32>, tensor<6x1xf32>
  }
}
