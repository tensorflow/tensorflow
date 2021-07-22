// RUN: tf-tfrt-opt -split-input-file -tf-outline-cpurt-cluster %s             \
// RUN: | FileCheck %s

// -----
// Outline a simple cluster with a single operation.

// CHECK-LABEL: func @simple_cluster
func @simple_cluster(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:      %[[RES:.*]] = cpurt.call(%arg0)
  // CHECK-SAME: {callee = @kernel::@compute}
  // CHECK-SAME: (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Rsqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
    tf_device.return %1 : tensor<?xf32>
  }) { policy = "tfrt.auto-fusion" } : () -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK:      module @kernel attributes {tfrt.compiled}
// CHECK:      func @compute(
// CHECK-SAME:   %arg0: tensor<?xf32>
// CHECK-SAME: ) -> tensor<?xf32> {
// CHECK:        %[[RET:.*]] = "tf.Rsqrt"(%arg0)
// CHECK:        return %[[RET]]
// CHECK:      }

// -----
// Check that tf.Transpose constraint propagated to the function argument.

// CHECK-LABEL: func @cluster_with_transpose
func @cluster_with_transpose(%arg0: tensor<?x?xf32>,
                             %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK:      %[[RES:.*]] = cpurt.call(%arg0, %arg1)
  // CHECK-SAME: {callee = @kernel::@compute}
  // CHECK-SAME: (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Transpose"(%arg0, %arg1)
         : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    tf_device.return %1 : tensor<?x?xf32>
  }) { policy = "tfrt.auto-fusion" } : () -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK:      module @kernel attributes {tfrt.compiled}
// CHECK:      func @compute(
// CHECK-SAME:   %arg0: tensor<?x?xf32>
// CHECK-SAME:   %arg1: tensor<2xi32> {cpurt.constraint = "value"}
// CHECK-SAME: ) -> tensor<?x?xf32> {
// CHECK:        %[[RET:.*]] = "tf.Transpose"(%arg0, %arg1)
// CHECK:        return %[[RET]]
// CHECK:      }
