// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:   --gml-st-cpu-tiling-pipeline=matmul-tile-sizes=4,5,6 | FileCheck %s

func.func @matvec(%lhs: tensor<33x17xf32>, %rhs: tensor<17xf32>,
                  %output: tensor<33xf32>) -> tensor<33xf32> {
  %2 = linalg.matvec ins(%lhs, %rhs : tensor<33x17xf32>, tensor<17xf32>)
                     outs(%output : tensor<33xf32>) -> tensor<33xf32>
  return %2 : tensor<33xf32>
}

// CHECK-LABEL: @matvec
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:     %[[C12:.*]] = arith.constant 12 : index
// CHECK-DAG:     %[[C17:.*]] = arith.constant 17 : index
// CHECK-DAG:     %[[C32:.*]] = arith.constant 32 : index
// CHECK:         scf.for {{.*}} %[[C0]] to %[[C32]] step %[[C4]]
// CHECK:           scf.for {{.*}} %[[C0]] to %[[C12]] step %[[C6]]
// CHECK:             vector.contract {{.*}} vector<4x6xf32>
// CHECK-NEXT:        scf.yield %{{.*}} : vector<4xf32>
// CHECK:           vector.contract
// CHECK:           vector.transfer_write
// CHECK:         scf.for {{.*}} %[[C0]] to %[[C17]] step %[[C6]]
// CHECK:           linalg.matvec

// -----

func.func @vecmat(%lhs: tensor<17xf32>, %rhs: tensor<17x33xf32>,
                  %output: tensor<33xf32>) -> tensor<33xf32> {
  %2 = linalg.vecmat ins(%lhs, %rhs : tensor<17xf32>, tensor<17x33xf32>)
                     outs(%output : tensor<33xf32>) -> tensor<33xf32>
  return %2 : tensor<33xf32>
}

// CHECK-LABEL: @vecmat
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:     %[[C12:.*]] = arith.constant 12 : index
// CHECK-DAG:     %[[C17:.*]] = arith.constant 17 : index
// CHECK-DAG:     %[[C30:.*]] = arith.constant 30 : index
// CHECK:         scf.for {{.*}} %[[C0]] to %[[C30]] step %[[C5]]
// CHECK:           scf.for {{.*}} %[[C0]] to %[[C12]] step %[[C6]]
// CHECK:             vector.contract {{.*}} vector<6x5xf32>
// CHECK-NEXT:        scf.yield %{{.*}} : vector<5xf32>
// CHECK:           vector.contract
// CHECK:           vector.transfer_write
// CHECK:         scf.for {{.*}} %[[C0]] to %[[C17]] step %[[C6]]
// CHECK:           linalg.vecmat

// -----

func.func @dot(%lhs: tensor<19xf32>, %rhs: tensor<19xf32>,
                  %output: tensor<f32>) -> tensor<f32> {
  %2 = linalg.dot ins(%lhs, %rhs : tensor<19xf32>, tensor<19xf32>)
                     outs(%output : tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

// CHECK-LABEL: @dot
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:     %[[C18:.*]] = arith.constant 18 : index
// CHECK:         scf.for {{.*}} %[[C0]] to %[[C18]] step %[[C6]]
// CHECK:           vector.contract {{.*}} vector<6xf32>
// CHECK-NEXT:      vector.broadcast
// CHECK-NEXT:      scf.yield %{{.*}} : vector<f32>
// CHECK:         arith.mulf
// CHECK:         arith.addf

// -----

func.func @matvec_to_vecmat(%rhs: tensor<2xi32>,
                            %output: tensor<3xi32>) -> tensor<3xi32> {
  %cst = arith.constant dense<[[0, 1], [2, 3], [4, 5]]> : tensor<3x2xi32>
  %2 = linalg.matvec ins(%cst, %rhs : tensor<3x2xi32>, tensor<2xi32>)
                     outs(%output : tensor<3xi32>) -> tensor<3xi32>
  return %2 : tensor<3xi32>
}

// CHECK-LABEL: @matvec_to_vecmat
// CHECK: arith.constant dense<{{\[}}[0, 2, 4], [1, 3, 5]]> : tensor<2x3xi32>
// CHECK: vector.contract {{.*}} : vector<2xi32>, vector<2x3xi32> into vector<3xi32>
