// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @insert_strided_slice() -> (vector<3x4xi32>, vector<3x4xi32>) {
  %v = arith.constant dense<[[2, 3, 4], [6, 7, 8]]> : vector<2x3xi32>
  %c = arith.constant dense<0> : vector<3x4xi32>
  %o = vector.insert_strided_slice %v, %c {
    offsets = [0, 1],
    // TODO(jreiffers): Test non-unit strides when supported by verifier.
    strides = [1, 1]
  } : vector<2x3xi32> into vector<3x4xi32>
  return %c, %o : vector<3x4xi32>, vector<3x4xi32>
}

// CHECK-LABEL: @insert_strided_slice
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
// CHECK-NEXT{LITERAL}: [[0, 2, 3, 4], [0, 6, 7, 8], [0, 0, 0, 0]]
