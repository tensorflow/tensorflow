// RUN: tf-tfrt-opt %s -pack-inputs="slices=1,0,24,2,24,24" | FileCheck %s

// Tests basic contiguous slice packing:
// - Input 0 is preserved (not in slice list).
// - Inputs 1 and 2 are packed into a single contiguous i8 buffer with explicit slices.

module {
  // CHECK-LABEL: func.func @main
  // CHECK-SAME: (%arg0: tensor<10x10xf32>, %arg1: tensor<48xi8>) -> (tensor<10x10xf32>, tensor<3x2xf32>, tensor<3x2xf32>)
  func.func @main(%arg0: tensor<10x10xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<3x2xf32>) -> (tensor<10x10xf32>, tensor<3x2xf32>, tensor<3x2xf32>) attributes {ifrt.function} {
    // CHECK: %[[SLICE0:.*]] = stablehlo.slice %arg1 [0:24] : (tensor<48xi8>) -> tensor<24xi8>
    // CHECK: %[[RESHAPE0:.*]] = stablehlo.reshape %[[SLICE0]] : (tensor<24xi8>) -> tensor<3x2x4xi8>
    // CHECK: %[[BITCAST0:.*]] = stablehlo.bitcast_convert %[[RESHAPE0]] : (tensor<3x2x4xi8>) -> tensor<3x2xf32>
    // CHECK: %[[SLICE1:.*]] = stablehlo.slice %arg1 [24:48] : (tensor<48xi8>) -> tensor<24xi8>
    // CHECK: %[[RESHAPE1:.*]] = stablehlo.reshape %[[SLICE1]] : (tensor<24xi8>) -> tensor<3x2x4xi8>
    // CHECK: %[[BITCAST1:.*]] = stablehlo.bitcast_convert %[[RESHAPE1]] : (tensor<3x2x4xi8>) -> tensor<3x2xf32>
    // CHECK: return %arg0, %[[BITCAST0]], %[[BITCAST1]] : tensor<10x10xf32>, tensor<3x2xf32>, tensor<3x2xf32>
    return %arg0, %arg1, %arg2 : tensor<10x10xf32>, tensor<3x2xf32>, tensor<3x2xf32>
  }
}
