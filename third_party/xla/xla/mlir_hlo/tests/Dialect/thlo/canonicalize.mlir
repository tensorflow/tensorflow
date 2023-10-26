// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --canonicalize | FileCheck %s

func.func @reverse_dynamic_fold(%input: tensor<1x?xf32>, %init: tensor<1x?xf32>)
  -> tensor<1x?xf32> {
  %res = thlo.reverse
         ins(%input: tensor<1x?xf32>)
         outs(%init: tensor<1x?xf32>)
         reverse_dimensions = [0]
  func.return %res : tensor<1x?xf32>
}

// CHECK-LABEL: func @reverse_dynamic_fold
//  CHECK-SAME: %[[ARG0:.*]]: tensor<1x?xf32>, %[[ARG1:.*]]: tensor<1x?xf32>
//       CHECK:   return %[[ARG0]]