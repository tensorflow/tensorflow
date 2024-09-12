// RUN: mlir-hlo-opt -mhlo-legalize-cross-replica-sum-to-all-reduce -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @cross_replica_sum_to_all_reduce
func.func @cross_replica_sum_to_all_reduce(%arg0 : tensor<4xi64>) -> tensor<4xi64> {
  // CHECK: [[RES:%.+]] = "mhlo.all_reduce"(%arg0)
  // CHECK-SAME{LITERAL}: <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
  // CHECK: ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
  // CHECK:   [[ADD:%.+]] = mhlo.add %arg1, %arg2 : tensor<i64>
  // CHECK:   mhlo.return [[ADD]] : tensor<i64>
  // CHECK-NEXT: }) : (tensor<4xi64>) -> tensor<4xi64>
  %0 = "mhlo.cross-replica-sum"(%arg0) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<4xi64>) -> tensor<4xi64>
  func.return %0 : tensor<4xi64>
}
