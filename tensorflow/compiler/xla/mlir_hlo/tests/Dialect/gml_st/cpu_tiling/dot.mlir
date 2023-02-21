// RUN: mlir-hlo-opt %s --split-input-file --gml-st-cpu-tiling-pipeline | FileCheck %s

func.func @matvec(%lhs: tensor<33x17xf32>, %rhs: tensor<17xf32>,
                  %output: tensor<33xf32>) -> tensor<33xf32> {
  %2 = linalg.matvec ins(%lhs, %rhs : tensor<33x17xf32>, tensor<17xf32>)
                     outs(%output : tensor<33xf32>) -> tensor<33xf32>
  return %2 : tensor<33xf32>
}

// CHECK-LABEL: @matvec
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             vector.extract {{.*}}[0] : vector<4xf32>
// CHECK:             vector.outerproduct {{.*}} vector<4xf32>, f32
// CHECK:             vector.extract {{.*}}[1] : vector<4xf32>
// CHECK:             vector.outerproduct {{.*}} vector<4xf32>, f32
// CHECK:             vector.extract {{.*}}[2] : vector<4xf32>
// CHECK:             vector.outerproduct {{.*}} vector<4xf32>, f32
// CHECK:             vector.extract {{.*}}[3] : vector<4xf32>
// CHECK:             vector.outerproduct {{.*}} vector<4xf32>, f32
// CHECK:           vector.outerproduct {{.*}} vector<4xf32>, f32
// CHECK:         scf.for
// CHECK:           linalg.matvec

// -----

func.func @vecmat(%lhs: tensor<17xf32>, %rhs: tensor<17x33xf32>,
                  %output: tensor<33xf32>) -> tensor<33xf32> {
  %2 = linalg.vecmat ins(%lhs, %rhs : tensor<17xf32>, tensor<17x33xf32>)
                     outs(%output : tensor<33xf32>) -> tensor<33xf32>
  return %2 : tensor<33xf32>
}

// CHECK-LABEL: @vecmat
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             vector.extract {{.*}}[0] : vector<4xf32>
// CHECK:             vector.outerproduct {{.*}} vector<4xf32>, f32
// CHECK:             vector.extract {{.*}}[1] : vector<4xf32>
// CHECK:             vector.outerproduct {{.*}} vector<4xf32>, f32
// CHECK:             vector.extract {{.*}}[2] : vector<4xf32>
// CHECK:             vector.outerproduct {{.*}} vector<4xf32>, f32
// CHECK:             vector.extract {{.*}}[3] : vector<4xf32>
// CHECK:             vector.outerproduct {{.*}} vector<4xf32>, f32
// CHECK:           vector.outerproduct {{.*}} vector<4xf32>, f32
// CHECK:         scf.for
// CHECK:           linalg.vecmat

// -----

func.func @dot(%lhs: tensor<17xf32>, %rhs: tensor<17xf32>,
                  %output: tensor<f32>) -> tensor<f32> {
  %2 = linalg.dot ins(%lhs, %rhs : tensor<17xf32>, tensor<17xf32>)
                     outs(%output : tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

// CHECK-LABEL: @dot
// CHECK:         scf.for
// CHECK:           %[[REDUCTION:.*]] = vector.reduction <add>
// CHECK-SAME:        vector<4xf32> into f32
// CHECK:           vector.broadcast %[[REDUCTION]] : f32 to vector<f32>
// CHECK:         arith.mulf
// CHECK:         arith.addf
