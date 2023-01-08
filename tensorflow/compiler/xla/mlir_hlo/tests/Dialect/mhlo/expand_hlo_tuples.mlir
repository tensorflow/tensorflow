// RUN: mlir-hlo-opt %s -split-input-file -expand-hlo-tuples='entry-function=main' | FileCheck %s
// Check if the `expand-hlo-tuples` pass adds the right variable to return_op and function return type.

func.func @main(%arg0: tensor<1x1xf32>, %arg1: tensor<1x8x8x16xf32>) -> tuple<tensor<1024xf32>, tensor<1xf32>> {
  %1 = "mhlo.reshape"(%arg0) : (tensor<1x1xf32>) -> tensor<1xf32>
  %2 = "mhlo.reshape"(%arg1) : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
  %3 = "mhlo.tuple"(%2, %1) {name = "tuple.374"} : (tensor<1024xf32>, tensor<1xf32>) -> tuple<tensor<1024xf32>, tensor<1xf32>>
  func.return %3 : tuple<tensor<1024xf32>, tensor<1xf32>>
  // CHECK: %[[RES0:.*]] = mhlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<1xf32>
  // CHECK: %[[RES1:.*]] = mhlo.reshape %arg1 : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
  // CHECK: return %[[RES1]], %[[RES0]] : tensor<1024xf32>, tensor<1xf32>
}

// -----
func.func @main(%arg0: tensor<1x224x224x3xf16>, %arg1: tensor<f32>) -> tensor<1x224x224x3xf16> {
  func.return %arg0 : tensor<1x224x224x3xf16>
}

// -----

func.func @main(%arg0: tuple<tensor<1024xf32>, tensor<1xf32>>) -> tuple<tensor<1024xf32>, tensor<1xf32>> {
  func.return %arg0 : tuple<tensor<1024xf32>, tensor<1xf32>>
}

// CHECK:   func @main(%[[VAL_0:.*]]: tensor<1024xf32>, %[[VAL_1:.*]]: tensor<1xf32>) -> (tensor<1024xf32>, tensor<1xf32>) {
// CHECK:           %[[VAL_2:.*]] = mhlo.tuple %[[VAL_0]], %[[VAL_1]] : tuple<tensor<1024xf32>, tensor<1xf32>>
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : tensor<1024xf32>, tensor<1xf32>
// CHECK:         }

// -----

func.func @main() -> tuple<> {
  %0 = "mhlo.tuple"() {xla_shape = "()"} : () -> tuple<>
  func.return %0 : tuple<>
}

// CHECK-LABEL: func @main() {
// CHECK:   return{{$}}
// CHECK:  }
