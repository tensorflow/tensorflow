// RUN: mlir-hlo-opt --stablehlo-ext-refine-shapes --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>) -> tensor<?x?xf32> {
  // CHECK: stablehlo.dynamic_reduce_window{{.*}} -> tensor<2x2xf32>
  %0 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[4, 1]> : tensor<2xi64>
  %2 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %3 = stablehlo.constant dense<[3, 1]> : tensor<2xi64>
  %4 = stablehlo.constant dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
  %5 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg0, %arg1, %0, %1, %2, %3, %4) {
    called_computations = [@dynamic_reduce_window0]
  } : (tensor<3x2xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<?x?xf32>
  func.return %5 : tensor<?x?xf32>
}

func.func private @dynamic_reduce_window0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @refine_dynamic_rng_bit_generator
func.func @refine_dynamic_rng_bit_generator(%arg0: tensor<2xui64>) -> (tensor<?xui64>, tensor<?x?xf32>) {
  // CHECK: stablehlo.dynamic_rng_bit_generator{{.*}} -> (tensor<2xui64>, tensor<1x4xf32>)
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_rng_bit_generator(%arg0, %0) {
    rng_algorithm = #stablehlo<rng_algorithm DEFAULT>
  } : (tensor<2xui64>, tensor<2xi64>) -> (tensor<?xui64>, tensor<?x?xf32>)
  func.return %1#0, %1#1 : tensor<?xui64>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @refine_dynamic_top_k
func.func @refine_dynamic_top_k(%arg0: tensor<16xf32>) -> (tensor<?xf32>, tensor<?xi32>) {
  // CHECK: stablehlo.dynamic_top_k{{.*}} -> (tensor<4xf32>, tensor<4xi32>)
  %k = stablehlo.constant dense<4> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<ui64>) -> (tensor<?xf32>, tensor<?xi32>)
  return %1#0, %1#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

// CHECK-LABEL: module @refine_call
module @refine_call {
  // CHECK: func.func @main{{.*}}-> (tensor<4xf32>, tensor<4xi32>)
  func.func @main(%arg1: tensor<16xf32>) -> (tensor<?xf32>, tensor<?xi32>) {
    %0 = stablehlo.bitcast_convert %arg1 : (tensor<16xf32>) -> tensor<?xf32>
    // CHECK: refine_call_callee{{.*}}-> (tensor<4xf32>, tensor<4xi32>)
    %2:2 = call @refine_call_callee(%0) : (tensor<?xf32>) -> (tensor<?xf32>, tensor<?xi32>)
    return %2#0, %2#1 : tensor<?xf32>, tensor<?xi32>
  }
  // CHECK: refine_call_callee(%arg0: tensor<16xf32>) -> (tensor<4xf32>, tensor<4xi32>)
  func.func @refine_call_callee(%arg0: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xi32>) {
    // CHECK: stablehlo.dynamic_top_k{{.*}} -> (tensor<4xf32>, tensor<4xi32>)
    %k = stablehlo.constant dense<4> : tensor<ui64>
    %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<?xf32>, tensor<ui64>) -> (tensor<?xf32>, tensor<?xi32>)
    return %1#0, %1#1 : tensor<?xf32>, tensor<?xi32>
  }
}
