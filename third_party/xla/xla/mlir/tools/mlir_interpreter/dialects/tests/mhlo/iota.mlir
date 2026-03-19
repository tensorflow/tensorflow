// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @iota_f32() -> tensor<1x2x3x4xf32> {
  %result = "mhlo.iota"() {
    iota_dimension = 2 : i64
  } : () -> tensor<1x2x3x4xf32>
  func.return %result : tensor<1x2x3x4xf32>
}

// CHECK-LABEL: @iota_f32
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
// CHECK{LITERAL}:         [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00],
// CHECK{LITERAL}:         [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]]]

func.func @iota_i32() -> tensor<1x2x3x4xi32> {
  %result = "mhlo.iota"() {
    iota_dimension = 3 : i64
  } : () -> tensor<1x2x3x4xi32>
  func.return %result : tensor<1x2x3x4xi32>
}

// CHECK-LABEL: @iota_i32
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[[0, 1, 2, 3],
// CHECK{LITERAL}          [0, 1, 2, 3],
// CHECK{LITERAL}          [0, 1, 2, 3]],
// CHECK{LITERAL}         [[0, 1, 2, 3],
// CHECK{LITERAL}          [0, 1, 2, 3],
// CHECK{LITERAL}          [0, 1, 2, 3]]]]
