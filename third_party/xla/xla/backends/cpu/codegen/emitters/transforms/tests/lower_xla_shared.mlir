// RUN: emitters_opt %s -split-input-file -xla-cpu-lower-xla-shared | FileCheck %s

func.func @forall_op(%input: tensor<1024x32x2xf32>) -> (tensor<1024x32x2xf32>) {
  %double = scf.forall (%i, %j, %k) in (1024, 32, 2)
      shared_outs(%captured_input = %input) -> (tensor<1024x32x2xf32>) {
    %t = tensor.extract %captured_input[%i, %j, %k] : tensor<1024x32x2xf32>
    %add = arith.addf %t, %t : f32
    %result = tensor.insert %add into %captured_input[%i, %j, %k]
        : tensor<1024x32x2xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %result into %captured_input[0, 0, 0][1024, 32, 2][1, 1, 1]
          : tensor<1024x32x2xf32> into tensor<1024x32x2xf32>
    }
  }
  func.return %double : tensor<1024x32x2xf32>
}
// CHECK-DAG: [[CONST_0:%.*]] = arith.constant 0 : index
// CHECK-DAG: [[CONST_1:%.*]] = arith.constant 1 : index
// CHECK-DAG: [[CONST_2:%.*]] = arith.constant 2 : index
// CHECK-DAG: [[CONST_32:%.*]] = arith.constant 32 : index
// CHECK-DAG: [[CONST_1024:%.*]] = arith.constant 1024 : index
// CHECK: [[FOR_ALL_0:%.*]] = scf.for [[IV_0:%.*]] = [[CONST_0]] to [[CONST_2]] step [[CONST_1]]
// CHECK: [[FOR_ALL_1:%.*]] = scf.for [[IV_1:%.*]] = [[CONST_0]] to [[CONST_32]] step [[CONST_1]]
// CHECK: [[FOR_ALL_2:%.*]] = scf.for [[IV_2:%.*]] = [[CONST_0]] to [[CONST_1024]] step [[CONST_1]]
// CHECK: tensor.extract {{%.*\[}}[[IV_2]], [[IV_1]], [[IV_0]]{{\]}}
