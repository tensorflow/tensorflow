// RUN: mlir-bisect %s --debug-strategy=ReplaceOpWithConstant | FileCheck %s

func.func @main() -> tensor<2xi32> {
  %a = arith.constant dense<3> : tensor<2xi32>
  %b = arith.constant dense<2> : tensor<2xi32>
  %c = mhlo.add %a, %b : tensor<2xi32>
  %d = mhlo.multiply %b, %c : tensor<2xi32>
  func.return %d : tensor<2xi32>
}

//      CHECK: func.func @main()
// CHECK-NEXT:   arith.constant dense<3>
// CHECK-NEXT:   arith.constant dense<2>
// CHECK-NEXT:   arith.constant dense<5>
// CHECK-NEXT:   %[[ADD:.*]] = mhlo.add
//  CHECK-NOT:   %[[ADD]]
// CHECK-NEXT:   mhlo.multiply
// CHECK-NEXT:   return

//      CHECK: func.func @main()
// CHECK-NEXT:   arith.constant dense<3>
// CHECK-NEXT:   arith.constant dense<2>
// CHECK-NEXT:   mhlo.add
// CHECK-NEXT:   %[[D:.*]] = arith.constant dense<10>
// CHECK-NEXT:   mhlo.multiply
// CHECK-NEXT:   return %[[D]]
