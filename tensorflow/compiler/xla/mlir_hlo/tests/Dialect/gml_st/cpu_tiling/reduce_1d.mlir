// RUN: mlir-hlo-opt %s --split-input-file --gml-st-cpu-tiling-pipeline \
// RUN: | FileCheck %s

func.func @reduce_1d_static(%arg0: tensor<100xf32>) -> tensor<f32> {
  %1 = tensor.empty() : tensor<f32>
  %cst = arith.constant 0.0 : f32
  %init = linalg.fill ins(%cst : f32) outs(%1 : tensor<f32>) -> tensor<f32>
  %res = linalg.reduce { arith.addf }
    ins(%arg0: tensor<100xf32>) outs(%init: tensor<f32>) dimensions = [0]
  return %res : tensor<f32>
}
// CHECK-LABEL: @reduce_1d_static

//       CHECK: arith.constant dense<0.000000e+00> : vector<8xf32>

//       CHECK: scf.for
//       CHECK:   vector.multi_reduction <add>
//  CHECK-SAME:     : vector<4x8xf32> to vector<8xf32>
//       CHECK:   scf.yield %{{.*}} : vector<8xf32>

//       CHECK: vector.multi_reduction <add>
//  CHECK-SAME:   : vector<8xf32> to f32
//       CHECK: vector.multi_reduction <add>
//  CHECK-SAME:   : vector<4xf32> to f32

// -----

func.func @reduce_1d_dynamic(%arg0: tensor<?xf32>) -> tensor<f32> {
  %1 = tensor.empty() : tensor<f32>
  %cst = arith.constant 0.0 : f32
  %init = linalg.fill ins(%cst : f32) outs(%1 : tensor<f32>) -> tensor<f32>
  %res = linalg.reduce { arith.addf }
    ins(%arg0: tensor<?xf32>) outs(%init: tensor<f32>) dimensions = [0]
  return %res : tensor<f32>
}
// CHECK-LABEL: func @reduce_1d_dynamic

//       CHECK: arith.constant dense<0.000000e+00> : vector<8xf32>

//       CHECK: scf.for
//       CHECK:   vector.multi_reduction <add>
//  CHECK-SAME:     : vector<4x8xf32> to vector<8xf32>
//       CHECK:   scf.yield %{{.*}} :  vector<8xf32>

//       CHECK: vector.multi_reduction <add>
//  CHECK-SAME:   : vector<8xf32> to f32

//       CHECK: scf.for
//       CHECK:   linalg.reduce
//       CHECK:   scf.yield %{{.*}} : tensor<f32>
