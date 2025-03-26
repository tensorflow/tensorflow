// RUN: mlir-interpreter-runner %s | FileCheck %s

func.func @main() -> tensor<2x3xi64> {
  %lhs = mhlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi64>
  %rhs = mhlo.constant dense<[[10, 20, 30], [40, 50, 60]]> : tensor<2x3xi64>
  %result = mhlo.subtract %lhs, %rhs : tensor<2x3xi64>
  return %result : tensor<2x3xi64>
}

// CHECK{LITERAL}: [[-10, -19, -28], [-37, -46, -55]]