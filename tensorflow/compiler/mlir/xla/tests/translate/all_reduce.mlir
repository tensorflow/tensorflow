// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "xla_hlo.all_reduce"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = xla_hlo.max %lhs, %rhs : tensor<f32>
    "xla_hlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_id = {
      handle = 5 : i64,
      type = 2 : i64
    }
  } : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK:  %[[COMPUTATION:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK-LABEL:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[10] all-reduce(f32[10] %[[ARG0]])
// CHECK-SAME:  channel_id=5
// CHECK-SAME:  replica_groups={{[{][{]}}0,2,4,6},{1,3,5,7{{[}][}]}}
// CHECK-SAME:  to_apply=%[[COMPUTATION]]
