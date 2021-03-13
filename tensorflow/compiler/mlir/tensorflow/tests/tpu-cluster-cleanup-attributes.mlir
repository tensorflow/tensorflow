// RUN: tf-opt %s -tf-tpu-cleanup-cluster-attributes | FileCheck %s

func @test(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) ->  tensor<f32> {
  // CHECK: "tf_device.cluster"
  // CHECK-NOT: _tpu_replicate =
  // CHECK-NOT: device =
  %1 = "tf_device.cluster"() ( {
    %2 = "tf.Add"(%arg1, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = "tf.IfRegion"(%arg0) ({
        %4 = "tf.Mul" (%arg1, %2) {device = "y"}: (tensor<f32>, tensor<f32>) -> tensor<f32>
        "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
        %5 = "tf.Div" (%arg1, %2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "tf.Yield"(%5) : (tensor<f32>) -> ()
      }) {is_stateless = true, _tpu_replicate = "x" } : (tensor<i1>) -> (tensor<f32>)
    tf_device.return %3 : tensor<f32>
  // CHECK: {_tpu_replicate = "x", cluster_attr = "cluster_attr", device = "y"}
  }) {cluster_attr = "cluster_attr", _tpu_replicate = "x", device = "y"} : () -> tensor<f32>
  // CHECK: "tf.Add"
  // CHECK-SAME: {_tpu_replicate = "x", device = "y"}
  %2 = "tf.Add"(%arg2, %1) {_tpu_replicate = "x", device = "y"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return
  return %2 : tensor<f32>
}
