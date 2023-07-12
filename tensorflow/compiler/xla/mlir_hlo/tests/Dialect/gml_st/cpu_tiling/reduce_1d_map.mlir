// RUN: mlir-hlo-opt %s \
// RUN: --gml-st-cpu-tiling-pipeline="reduction-1d-tile-size=32 reduction-1d-split-ratio=8" \
// RUN: | FileCheck %s
func.func @reduce_1d_map_aka_dot(%lhs: tensor<?xf32>,
    %rhs: tensor<?xf32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %size = tensor.dim %lhs, %c0 : tensor<?xf32>
  %init_1d = tensor.empty(%size) : tensor<?xf32>

  %map = linalg.map { arith.mulf }
    ins(%lhs, %rhs: tensor<?xf32>, tensor<?xf32>) outs(%init_1d: tensor<?xf32>)
  %cst = arith.constant 0.0 : f32
  %init_0d = tensor.empty() : tensor<f32>

  %fill = linalg.fill
    ins(%cst : f32) outs(%init_0d : tensor<f32>) -> tensor<f32>
  %res = linalg.reduce { arith.addf }
    ins(%map: tensor<?xf32>) outs(%fill: tensor<f32>) dimensions = [0]
  return %res : tensor<f32>
}
// CHECK-LABEL: func.func @reduce_1d_map_aka_dot
// CHECK: scf.for
// CHECK:   arith.mulf {{.*}} : vector<32xf32>
// CHECK:   vector.multi_reduction <add>
// CHECK:     : vector<4x8xf32> to vector<8xf32>
// CHECK:   scf.yield %{{.*}} : vector<8xf32>
// CHECK: vector.multi_reduction <add>
// CHECK:   : vector<8xf32> to f32
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     arith.mulf {{.*}} : f32
// CHECK:     arith.addf {{.*}} : f32
// CHECK:     scf.yield {{.*}} : f32
// CHECK:   scf.yield {{.*}} : f32

