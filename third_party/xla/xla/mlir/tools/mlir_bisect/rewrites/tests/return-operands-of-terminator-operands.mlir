// RUN: mlir-bisect %s --debug-strategy=ReturnOperandsOfTerminatorOperands | FileCheck %s

func.func @main() -> tensor<2xi32> {
  %a = arith.constant dense<3> : tensor<2xi32>
  %b = arith.constant dense<2> : tensor<2xi32>
  %c = mhlo.add %a, %b : tensor<2xi32>
  %d = mhlo.multiply %b, %c : tensor<2xi32>
  func.return %d : tensor<2xi32>
}

// CHECK: @main
// CHECK:   %[[C2:.*]] = arith.constant dense<2>
// CHECK:   %[[ADD:.*]] = mhlo.add
// CHECK:   mhlo.multiply
// CHECK:   return %[[C2]], %[[ADD]]