// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = xla_hlo.constant dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi32>
  %1 = "xla_hlo.cross-replica-sum"(%arg0) {replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>} : (tensor<10xf32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK: %[[SUM_COMPUTATION:.*]] ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[]) -> f32[]
// CHECK:  ROOT %[[RESULT:.*]] = f32[] add(f32[] %[[ARG0]], f32[] %[[ARG1]])

// CHECK: ENTRY %main
// CHECK: %[[ARG0:.*]] = f32[10] parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = f32[10] all-reduce(f32[10] %[[ARG0]])
// CHECK-SAME: replica_groups={{[{][{]}}0,2,4,6},{1,3,5,7{{[}][}]}}
// CHECK-SAME: to_apply=%[[SUM_COMPUTATION]]
