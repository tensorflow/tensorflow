// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @multi_reduction_1d_0d() -> i32 {
  %a = arith.constant dense<[10, 4, 7]> : vector<3xi32>
  %acc = arith.constant 1 : i32
  %r = vector.multi_reduction <add>, %a, %acc [0] : vector<3xi32> to i32
  return %r : i32
}

// CHECK-LABEL: @multi_reduction_1d_0d
// CHECK-NEXT: Results
// CHECK-NEXT: 22

#dense234 = dense<[[[0,1,2,3],[4,5,6,7],[8,9,10,11]],
                   [[12,13,14,15],[16,17,18,19],[20,21,22,23]]]>
              : vector<2x3x4xi32>

func.func @multi_reduction_3d_2d()
    -> (vector<2x3xi32>, vector<2x4xi32>, vector<2x3xi32>) {
  %a = arith.constant #dense234
  %acc_23 = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : vector<2x3xi32>
  %r1 = vector.multi_reduction <add>, %a, %acc_23 [2]
    : vector<2x3x4xi32> to vector<2x3xi32>
  %acc_24 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : vector<2x4xi32>
  %r2 = vector.multi_reduction <mul>, %a, %acc_24 [1]
    : vector<2x3x4xi32> to vector<2x4xi32>
  return %r1, %r2, %acc_23 : vector<2x3xi32>, vector<2x4xi32>, vector<2x3xi32>
}

// CHECK-LABEL: @multi_reduction_3d_2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[6, 23, 40], [57, 74, 91]]
// CHECK-NEXT{LITERAL}: [[0, 45, 240, 693], [15360, 23205, 33264, 45885]]
// CHECK-NEXT{LITERAL}: [[0, 1, 2], [3, 4, 5]]

func.func @multi_reduction_3d_1d() -> vector<3xi32> {
  %a = arith.constant #dense234
  %acc_3 = arith.constant dense<[0, 1, 2]> : vector<3xi32>
  %r1 = vector.multi_reduction <add>, %a, %acc_3 [2, 0]
    : vector<2x3x4xi32> to vector<3xi32>
  return %r1 : vector<3xi32>
}

// CHECK-LABEL: @multi_reduction_3d_1d
// CHECK-NEXT: Results
// CHECK-NEXT: [60, 93, 126]
