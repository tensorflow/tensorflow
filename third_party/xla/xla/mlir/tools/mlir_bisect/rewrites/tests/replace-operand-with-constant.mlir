// RUN: mlir-bisect %s --debug-strategy=ReplaceOperandWithConstant | FileCheck %s

func.func @main() -> (tensor<2xi32>, tensor<2xi32>) {
  %a = arith.constant dense<3> : tensor<2xi32>
  %b = arith.constant dense<2> : tensor<2xi32>
  %c = mhlo.add %a, %b : tensor<2xi32>
  %d = mhlo.multiply %b, %c : tensor<2xi32>
  func.return %c, %d : tensor<2xi32>, tensor<2xi32>
}

// CHECK: func @main()
// CHECK:   %[[C2:.*]] = arith.constant dense<2>
// CHECK:   %[[ADD:.*]] = mhlo.add
// CHECK:   %[[C5:.*]] = arith.constant dense<5>
// CHECK:   %[[MUL:.*]] = mhlo.multiply %[[C2]], %[[C5]] : tensor<2xi32>
// CHECK:   return %[[ADD]], %[[MUL]]

// CHECK: func @main()
// CHECK:   mhlo.add
// CHECK:   %[[MUL:.*]] = mhlo.multiply %cst_0, %0 : tensor<2xi32>
// CHECK:   %[[C5:.*]] = arith.constant dense<5>
// CHECK:   return %[[C5]], %[[MUL]]

// CHECK: func @main()
// CHECK:   %[[ADD:.*]] = mhlo.add
// CHECK:   mhlo.multiply
// CHECK:   %[[C10:.*]] = arith.constant dense<10>
// CHECK:   return %[[ADD]], %[[C10]]
