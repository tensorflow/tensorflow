// RUN: emitters_opt %s -split-input-file -xla-gpu-unswitch-loops | FileCheck %s

module {
  func.func @unswitchable(
     %arg0: tensor<2xf32>,
     %arg1: index
  ) -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst3 = arith.constant 3.0 : f32
    %cst4 = arith.constant 4.0 : f32
    %cond = arith.cmpi sle, %arg1, %c1 : index

    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
      %result = scf.if %cond -> tensor<2xf32> {
        %set_3 = tensor.insert %cst3 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_3 : tensor<2xf32>
      } else {
        %set_4 = tensor.insert %cst4 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_4 : tensor<2xf32>
      }
      scf.yield %result : tensor<2xf32>
    }

    func.return %for : tensor<2xf32>
  }
}

// CHECK:      @unswitchable(%[[ARG0:.*]]: tensor<2xf32>, %[[ARG1:.*]]: index)
// CHECK:        %[[CST3:.*]] = arith.constant 3.0
// CHECK:        %[[CST4:.*]] = arith.constant 4.0
// CHECK:        %[[COND:.*]] = arith.cmpi sle, %[[ARG1]]
// CHECK-NEXT:   scf.if %[[COND]]
// CHECK-NEXT:     scf.for
// CHECK-NEXT:       tensor.insert %[[CST3]]
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK:        } else {
// CHECK-NEXT:     scf.for
// CHECK-NEXT:       tensor.insert %[[CST4]]
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:   }
