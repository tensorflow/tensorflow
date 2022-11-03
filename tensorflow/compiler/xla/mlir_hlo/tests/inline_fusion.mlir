// RUN: mlir-hlo-opt --inline-fusion %s | FileCheck %s

// CHECK-LABEL: func @fusion
func.func @fusion(%arg0: tensor<8xf32>) -> tensor<8xf32> {

  // CHECK-NEXT: %[[EXP:.*]] = mhlo.exponential %arg0
  %0 = "mhlo.fusion"(%arg0) ({
  ^bb0(%arg1: tensor<8xf32>):
    %1 = mhlo.exponential %arg1 : tensor<8xf32>
    mhlo.return %1 : tensor<8xf32>
  }) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK-NEXT: return %[[EXP]] : tensor<8xf32>
  return %0 : tensor<8xf32>
}
