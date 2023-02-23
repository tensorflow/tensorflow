// RUN: mlir-hlo-opt %s --rewrite-vector-multi-reduction | FileCheck %s

// CHECK-LABEL: func @vector_row
func.func @vector_row(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    func.return %0 : vector<2xf32>
}
// CHECK-COUNT-4: arith.mulf

// CHECK-LABEL: func @vector_col
func.func @vector_col(%arg0: vector<2x4xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0] : vector<2x4xf32> to vector<4xf32>
    func.return %0 : vector<4xf32>
}
// CHECK: arith.mulf
// CHECK: arith.mulf

// CHECK-LABEL: func @vector_1d
func.func @vector_1d(%arg0: vector<4xf32>, %acc: f32) -> f32 {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0] : vector<4xf32> to f32
    func.return %0 : f32
}
// CHECK: vector.reduction <mul>
