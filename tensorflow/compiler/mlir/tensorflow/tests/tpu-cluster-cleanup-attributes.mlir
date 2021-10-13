// RUN: tf-opt %s -tf-tpu-cleanup-cluster-attributes | FileCheck %s

// CHECK-LABEL: func @control_flow_cleanup
func @control_flow_cleanup(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) ->  tensor<f32> {
  // CHECK: "tf_device.cluster"
  // CHECK-NOT: _tpu_replicate =
  // CHECK-NOT: device =
  %1 = "tf_device.cluster"() ( {
    %2 = "tf.Add"(%arg1, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = "tf.IfRegion"(%arg0) ({
        %4 = "tf.Mul" (%arg1, %2) {device = "y"}: (tensor<f32>, tensor<f32>) -> tensor<f32>
        "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
        // CHECK: device = "/device:TPU_REPLICATED_CORE:0"
        %5 = "tf.Div" (%arg1, %2) {device = "/device:TPU_REPLICATED_CORE:0"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
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

// CHECK-LABEL: func @skip_launch_device
func @skip_launch_device(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) ->  tensor<f32> {
  // CHECK: "tf_device.cluster"
  // CHECK: "tf_device.launch"
  // CHECK-NOT: _tpu_replicate =
  // CHECK: device = "y"
  %1 = "tf_device.cluster"() ( {
    %2 = "tf_device.launch"() ( {
      %3 = "tf.Add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      tf_device.return %3 : tensor<f32>
    }) {_tpu_replicate = "x", device = "y"} : () -> tensor<f32>
    tf_device.return %2 : tensor<f32>
  }) {cluster_attr = "cluster_attr", _tpu_replicate = "x", device = "y"} : () -> tensor<f32>

  return %1 : tensor<f32>
}

// CHECK-LABEL: func @remove_class_attribute
func @remove_class_attribute(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) ->  tensor<f32> {
  // CHECK: "tf_device.cluster"
  // CHECK: "tf.Add"
  // CHECK-NOT: _class
  %1 = "tf_device.cluster"() ( {
    %2 = "tf.Add"(%arg1, %arg2) {_class = ["loc:@while/BiasAdd_21/handle"]} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tf_device.return %2 : tensor<f32>
  }) {cluster_attr = "cluster_attr", _tpu_replicate = "x", device = "y"} : () -> tensor<f32>

  return %1 : tensor<f32>
}
