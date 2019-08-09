// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: extractelement
func @extractelement(%arg0: vector<4x8x16xf32>) -> (vector<8x16xf32>, vector<16xf32>, f32) {
  //      CHECK: vector.extractelement {{.*}}[3 : i32] : vector<4x8x16xf32>
  %1 = vector.extractelement %arg0[3 : i32] : vector<4x8x16xf32>
  // CHECK-NEXT: vector.extractelement {{.*}}[3 : i32, 3 : i32] : vector<4x8x16xf32>
  %2 = vector.extractelement %arg0[3 : i32, 3 : i32] : vector<4x8x16xf32>
  // CHECK-NEXT: vector.extractelement {{.*}}[3 : i32, 3 : i32, 3 : i32] : vector<4x8x16xf32>
  %3 = vector.extractelement %arg0[3 : i32, 3 : i32, 3 : i32] : vector<4x8x16xf32>
  return %1, %2, %3 : vector<8x16xf32>, vector<16xf32>, f32
}