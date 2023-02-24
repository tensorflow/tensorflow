// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @bitcast_convert() -> tensor<3xf32> {
  %c = arith.constant dense<[10000000,20000000,30000000]> : tensor<3xi32>
  %ret = mhlo.bitcast_convert %c : (tensor<3xi32>) -> tensor<3xf32>
  return %ret : tensor<3xf32>
}

// CHECK-LABEL: @bitcast_convert
// CHECK-NEXT: Results
// CHECK-NEXT: [1.401298e-38, 3.254205e-38, 7.411627e-38]
