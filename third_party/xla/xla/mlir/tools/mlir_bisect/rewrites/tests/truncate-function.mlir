// RUN: mlir-bisect %s --debug-strategy=TruncateFunction | FileCheck %s

// Function to prevent constant folding below.
func.func private @cst() -> tensor<2xi32> {
  %cst = arith.constant dense<2> : tensor<2xi32>
  return %cst : tensor<2xi32>
}

func.func @main() -> tensor<2xi32> {
  %a = arith.constant dense<1> : tensor<2xi32>
  %b = func.call @cst() : () -> tensor<2xi32>
  %c = mhlo.add %a, %b : tensor<2xi32>
  %d = mhlo.multiply %b, %c : tensor<2xi32>
  func.return %d : tensor<2xi32>
}

//     CHECK: func @main()
//     CHECK:   %[[A:.*]] = arith.constant dense<1>
//     CHECK:   return %[[A]]

//     CHECK: func @main()
//     CHECK:   %[[B:.*]] = call @cst()
//     CHECK:   return %[[B]]

//     CHECK: func @main()
//     CHECK:   %[[A:.*]] = arith.constant dense<1>
//     CHECK:   %[[B:.*]] = call @cst()
//     CHECK:   %[[ADD:.*]] = mhlo.add
// CHECK-DAG:   %[[A]]
// CHECK-DAG:   %[[B]]
//     CHECK:   return %[[ADD]]
