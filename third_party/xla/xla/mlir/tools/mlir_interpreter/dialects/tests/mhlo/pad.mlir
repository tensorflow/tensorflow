// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @static_pad() -> tensor<2x4x7xi32> {
  // lax.pad(np.array([[[1,2,3],[4,5,6]]]), 42,
  //                  [(0, 1, 0), (1, 1, 0), (2, 0, 1)])
  %cst = mhlo.constant dense<[[[1,2,3],[4,5,6]]]> : tensor<1x2x3xi32>
  %pad_value = mhlo.constant dense<42> : tensor<i32>
  %0 = "mhlo.pad"(%cst, %pad_value) {
    edge_padding_low = dense<[0, 1, 2]> : tensor<3xi64>,
    edge_padding_high = dense<[1, 1, 0]> : tensor<3xi64>,
    interior_padding = dense<[0, 0, 1]> : tensor<3xi64>
  } : (tensor<1x2x3xi32>, tensor<i32>) -> tensor<2x4x7xi32>
  func.return %0 : tensor<2x4x7xi32>
}

// CHECK-LABEL: @static_pad
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[42, 42, 42, 42, 42, 42, 42],
// CHECK-SAME{LITERAL}:   [42, 42, 1, 42, 2, 42, 3]
// CHECK-SAME{LITERAL}:   [42, 42, 4, 42, 5, 42, 6],
// CHECK-SAME{LITERAL}:   [42, 42, 42, 42, 42, 42, 42]],
// CHECK-SAME{LITERAL}:  [[42, 42, 42, 42, 42, 42, 42],
// CHECK-SAME{LITERAL}:   [42, 42, 42, 42, 42, 42, 42],
// CHECK-SAME{LITERAL}:   [42, 42, 42, 42, 42, 42, 42],
// CHECK-SAME{LITERAL}:   [42, 42, 42, 42, 42, 42, 42]]]

func.func @dynamic_pad() -> tensor<?x4xi32> {
  %c1 = arith.constant 1 : index
  %empty = tensor.empty(%c1) : tensor<?x2xi32>
  %pad_value = mhlo.constant dense<42> : tensor<i32>
  %0 = "mhlo.pad"(%empty, %pad_value) {
    edge_padding_low = dense<[0, 1]> : tensor<2xi64>,
    edge_padding_high = dense<[1, 1]> : tensor<2xi64>,
    interior_padding = dense<[0, 0]> : tensor<2xi64>
  } : (tensor<?x2xi32>, tensor<i32>) -> tensor<?x4xi32>
  func.return %0 : tensor<?x4xi32>
}

// CHECK-LABEL: @dynamic_pad
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[42, 0, 0, 42], [42, 42, 42, 42]]

func.func @negative_pad() -> tensor<10xi32> {
  %empty = arith.constant dense<[1,2,3,4,5,6,7]> : tensor<7xi32>
  %pad_value = mhlo.constant dense<42> : tensor<i32>
  %0 = "mhlo.pad"(%empty, %pad_value) {
    edge_padding_low = dense<[-2]> : tensor<1xi64>,
    edge_padding_high = dense<[-1]> : tensor<1xi64>,
    interior_padding = dense<[1]> : tensor<1xi64>
  } : (tensor<7xi32>, tensor<i32>) -> tensor<10xi32>
  func.return %0 : tensor<10xi32>
}

// CHECK-LABEL: @negative_pad
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [2, 42, 3, 42, 4, 42, 5, 42, 6, 42]
