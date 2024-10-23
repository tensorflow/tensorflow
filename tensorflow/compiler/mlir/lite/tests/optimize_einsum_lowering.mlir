// RUN: tf-opt -tfl-optimize -split-input-file %s | FileCheck %s

// CHECK-LABEL: FoldRedundantTransposeIntoReshape
func.func @FoldRedundantTransposeIntoReshape(%arg0: tensor<8x256x1792xf32>, %arg1: tensor<1x128x8x256xf32>) -> (tensor<128x1792xf32>){
  %cst_0 = arith.constant dense<[2048, 1792]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[128, 2048]> : tensor<2xi32>
  %cst_2 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
  %cst_3 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg1, %cst_3) : (tensor<1x128x8x256xf32>, tensor<4xi32>) -> tensor<1x128x256x8xf32>
  %1 = "tfl.transpose"(%arg0, %cst_2) : (tensor<8x256x1792xf32>, tensor<3xi32>) -> tensor<256x8x1792xf32>
  %2 = "tfl.reshape"(%0, %cst_1) : (tensor<1x128x256x8xf32>, tensor<2xi32>) -> tensor<128x2048xf32>
  %3 = "tfl.reshape"(%1, %cst_0) : (tensor<256x8x1792xf32>, tensor<2xi32>) -> tensor<2048x1792xf32>
  %4 = "tfl.batch_matmul"(%2, %3) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<128x2048xf32>, tensor<2048x1792xf32>) -> tensor<128x1792xf32>
  return %4 : tensor<128x1792xf32>
  // CHECK:  %cst = arith.constant dense<[2048, 1792]> : tensor<2xi32>
  // CHECK:  %cst_0 = arith.constant dense<[128, 2048]> : tensor<2xi32>
  // CHECK:  %0 = "tfl.reshape"(%arg1, %cst_0) : (tensor<1x128x8x256xf32>, tensor<2xi32>) -> tensor<128x2048xf32>
  // CHECK:  %1 = "tfl.reshape"(%arg0, %cst) : (tensor<8x256x1792xf32>, tensor<2xi32>) -> tensor<2048x1792xf32>
  // CHECK:  %2 = "tfl.batch_matmul"(%0, %1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<128x2048xf32>, tensor<2048x1792xf32>) -> tensor<128x1792xf32>
  // CHECK:  return %2 : tensor<128x1792xf32>
}

// CHECK-LABEL: FoldRedundantTransposeIntoReshape_ExpandedInputDims
func.func @FoldRedundantTransposeIntoReshape_ExpandedInputDims(%arg0: tensor<8x256x1792xf32>, %arg1: tensor<1x1x8x256xf32>) -> (tensor<1x1792xf32>) {
  %cst_0 = arith.constant dense<[2048, 1792]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[1, 2048]> : tensor<2xi32>
  %cst_2 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
  %cst_3 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg1, %cst_3) : (tensor<1x1x8x256xf32>, tensor<4xi32>) -> tensor<1x1x256x8xf32>
  %1 = "tfl.transpose"(%arg0, %cst_2) : (tensor<8x256x1792xf32>, tensor<3xi32>) -> tensor<256x8x1792xf32>
  %2 = "tfl.reshape"(%0, %cst_1) : (tensor<1x1x256x8xf32>, tensor<2xi32>) -> tensor<1x2048xf32>
  %3 = "tfl.reshape"(%1, %cst_0) : (tensor<256x8x1792xf32>, tensor<2xi32>) -> tensor<2048x1792xf32>
  %4 = "tfl.batch_matmul"(%2, %3) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<1x2048xf32>, tensor<2048x1792xf32>) -> tensor<1x1792xf32>
  return %4 : tensor<1x1792xf32>
  // CHECK:  %cst = arith.constant dense<[2048, 1792]> : tensor<2xi32>
  // CHECK:  %cst_0 = arith.constant dense<[1, 2048]> : tensor<2xi32>
  // CHECK:  %0 = "tfl.reshape"(%arg1, %cst_0) : (tensor<1x1x8x256xf32>, tensor<2xi32>) -> tensor<1x2048xf32>
  // CHECK:  %1 = "tfl.reshape"(%arg0, %cst) : (tensor<8x256x1792xf32>, tensor<2xi32>) -> tensor<2048x1792xf32>
  // CHECK:  %2 = "tfl.batch_matmul"(%0, %1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<1x2048xf32>, tensor<2048x1792xf32>) -> tensor<1x1792xf32>
  // CHECK:  return %2 : tensor<1x1792xf32>
}