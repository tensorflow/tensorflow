// RUN: emitters_opt %s -split-input-file -xla-propagate-alias-scopes | FileCheck %s

func.func @nested_for(
    %arg0: tensor<8x128xf32> {xla.invariant, xla.slice_index = 0 : index},
    %arg1: tensor<128x8xf32> {xla.slice_index = 1 : index}) -> tensor<128x8xf32>
    attributes { xla.entry }
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %0 = scf.for %arg2 = %c0 to %c128 step %c1
      iter_args(%arg3 = %arg1) -> (tensor<128x8xf32>) {
    %1 = scf.for %arg4 = %c0 to %c8 step %c1
        iter_args(%arg5 = %arg3) -> (tensor<128x8xf32>) {
      %extracted = tensor.extract %arg0[%arg4, %arg2] : tensor<8x128xf32>
      %inserted = tensor.insert %extracted into
        %arg5[%arg2, %arg4] : tensor<128x8xf32>
      scf.yield %inserted : tensor<128x8xf32>
    }
    scf.yield %1 : tensor<128x8xf32>
  }
  func.return %0 : tensor<128x8xf32>
}
// CHECK-LABEL: func.func @nested_for
// CHECK: tensor.extract {{.*}}llvm.noalias = [#[[ALIAS_SCOPE:[a-z0-9_]+]]
// CHECK: tensor.insert {{.*}}alias_scopes = [#[[ALIAS_SCOPE]]

// -----

func.func @multi_output(
    %arg0: tensor<8x128xf32> {xla.invariant, xla.slice_index = 0 : index},
    %arg1: tensor<128x8xf32> {xla.slice_index = 1 : index},
    %arg2: tensor<128x8xf32> {xla.slice_index = 2 : index}
  ) -> (tensor<128x8xf32>, tensor<128x8xf32>) attributes { xla.entry }
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %outer_res_0, %outer_res_1 = scf.for %outer_idx = %c0 to %c128 step %c1
      iter_args(%arg1_0 = %arg1, %arg2_0 = %arg2)
      -> (tensor<128x8xf32>, tensor<128x8xf32>) {
    %inner_res_0, %inner_res_1 = scf.for %inner_idx = %c0 to %c8 step %c1
        iter_args(%arg1_1 = %arg1_0, %arg2_1 = %arg2_0)
        -> (tensor<128x8xf32>, tensor<128x8xf32>) {
      %extracted = tensor.extract %arg0[%inner_idx, %outer_idx] : tensor<8x128xf32>
      %inserted_0 = tensor.insert %extracted into
        %arg1_1[%outer_idx, %inner_idx] : tensor<128x8xf32>
      %inserted_1 = tensor.insert %extracted into
        %arg2_1[%outer_idx, %inner_idx] : tensor<128x8xf32>
      scf.yield %inserted_0, %inserted_1 : tensor<128x8xf32>, tensor<128x8xf32>
    }
    scf.yield %inner_res_0, %inner_res_1 : tensor<128x8xf32>, tensor<128x8xf32>
  }
  func.return %outer_res_0, %outer_res_1 : tensor<128x8xf32>, tensor<128x8xf32>
}
// CHECK-LABEL: func.func @multi_output
// CHECK: tensor.extract {{.*}}llvm.noalias = [#[[ALIAS_SCOPE_1:[a-z0-9_]+]],
// CHECK-SAME: #[[ALIAS_SCOPE_2:[a-z0-9_]+]]
// CHECK-DAG : tensor.insert {{.*}}alias_scopes = [#[[ALIAS_SCOPE_1]]],
// CHECK-DAG-SAME: llvm.noalias = [#[[ALIAS_SCOPE_2]]]
// CHECK-DAG : tensor.insert {{.*}}alias_scopes = [#[[ALIAS_SCOPE_2]]],
// CHECK-DAG-SAME: llvm.noalias = [#[[ALIAS_SCOPE_1]]]

// -----

func.func @sub_call(
    %arg0: tensor<128xf32> {xla.invariant, xla.slice_index = 0 : index},
    %arg1: tensor<128xf32> {xla.slice_index = 1 : index}) -> tensor<128xf32>
    attributes { xla.entry }
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %0 = scf.for %index = %c0 to %c128 step %c1
      iter_args(%arg3 = %arg1) -> (tensor<128xf32>) {
    %result = xla.pure_call @sub_call_sub(%index, %arg0, %arg1)
      : (index, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    scf.yield %result : tensor<128xf32>
  }
  func.return %0 : tensor<128xf32>
}

func.func @sub_call_sub(
    %index: index, %arg0: tensor<128xf32>, %arg1: tensor<128xf32>
  ) -> tensor<128xf32>
{
  %extracted = tensor.extract %arg0[%index] : tensor<128xf32>
  %inserted = tensor.insert %extracted into %arg1[%index] : tensor<128xf32>
  func.return %inserted : tensor<128xf32>
}

// CHECK-LABEL: func.func @sub_call
// CHECK-LABEL: func.func @sub_call_sub
// CHECK: tensor.extract {{.*}}llvm.noalias = [#[[ALIAS_SCOPE:[a-z0-9_]+]]
// CHECK: tensor.insert {{.*}}alias_scopes = [#[[ALIAS_SCOPE]]
