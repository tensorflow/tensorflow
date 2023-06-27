// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @main() -> (tensor<2x3xi32>, tensor<2x3xf32>, tensor<0x0x3xi16>, tensor<f64>) {
  %i32 = mhlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %f32 = mhlo.constant dense<[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]> : tensor<2x3xf32>
  %empty = mhlo.constant dense<> : tensor<0x0x3xi16>
  %scalar = mhlo.constant dense<3.14> : tensor<f64>
  return %i32, %f32, %empty, %scalar : tensor<2x3xi32>, tensor<2x3xf32>, tensor<0x0x3xi16>, tensor<f64>
}

// CHECK-LABEL: @main
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 1, 2], [3, 4, 5]]
// CHECK-NEXT{LITERAL}: [[0.000000e+00, 1.000000e-01, 2.000000e-01], [3.000000e-01, 4.000000e-01, 5.000000e-01]]
// CHECK-NEXT{LITERAL}: []
// CHECK-NEXT{LITERAL}: 3.140000e+00

func.func @ui8() -> tensor<ui8> {
  %v = mhlo.constant dense<123> : tensor<ui8>
  return %v : tensor<ui8>
}

// CHECK-LABEL: @ui8
// CHECK-NEXT: Results
// CHECK-NEXT: 123