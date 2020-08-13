// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-launch-to-device-attribute | FileCheck %s


// Tests single TensorFlow op is hoisted out and has the correct device assigned
// by parent `tf_device.launch`.
// CHECK-LABEL: func @single_op_launch
func @single_op_launch() {
  tf_executor.graph {
    %0:5 = tf_executor.island {
      %a = "tf.opA"() : () -> tensor<i1>
      %launch:2 = "tf_device.launch"() ( {
        %b:2 = "tf.opB"(%a) : (tensor<i1>) -> (tensor<i32>, tensor<f32>)
        tf_device.return %b#1, %b#0 : tensor<f32>, tensor<i32>
      }) {device = "CPU:0"} : () -> (tensor<f32>, tensor<i32>)
      %c = "tf.opC"() : () -> tensor<i1>
      tf_executor.yield %a, %launch#0, %launch#1, %c : tensor<i1>, tensor<f32>, tensor<i32>, tensor<i1>
    }
    tf_executor.fetch
  }
  return
}

// CHECK:      %[[A:.*]] = "tf.opA"
// CHECK:      %[[B:.*]]:2 = "tf.opB"(%[[A]])
// CHECK-SAME: device = "CPU:0"
// CHECK:      %[[C:.*]] = "tf.opC"
// CHECK-NOT:  "tf_device.launch"
// CHECK:      tf_executor.yield %[[A]], %[[B]]#1, %[[B]]#0, %[[C]]


// Tests multiple TensorFlow ops are hoisted out and all have the correct device
// assigned by parent `tf_device.launch`.
// CHECK-LABEL: func @multi_op_launch
func @multi_op_launch() {
  tf_executor.graph {
    %0:5 = tf_executor.island {
      %a = "tf.opA"() : () -> tensor<i1>
      %launch:2 = "tf_device.launch"() ( {
        %b = "tf.opB"(%a) : (tensor<i1>) -> tensor<i32>
        %c = "tf.opC"(%b) : (tensor<i32>) -> tensor<f32>
        tf_device.return %c, %b : tensor<f32>, tensor<i32>
      }) {device = "CPU:0"} : () -> (tensor<f32>, tensor<i32>)
      %d = "tf.opD"() : () -> tensor<i1>
      tf_executor.yield %a, %launch#0, %launch#1, %d : tensor<i1>, tensor<f32>, tensor<i32>, tensor<i1>
    }
    tf_executor.fetch
  }
  return
}

// CHECK:      %[[A:.*]] = "tf.opA"
// CHECK:      %[[B:.*]] = "tf.opB"(%[[A]])
// CHECK-SAME: device = "CPU:0"
// CHECK:      %[[C:.*]] = "tf.opC"(%[[B]])
// CHECK-SAME: device = "CPU:0"
// CHECK:      %[[D:.*]] = "tf.opD"
// CHECK-NOT:  "tf_device.launch"
// CHECK:      tf_executor.yield %[[A]], %[[C]], %[[B]], %[[D]]


// -----


// Tests ops are hoisted out and devices are set only if the `tf_device.launch`
// contains TensorFlow ops.
func @non_tf_dialect_op_launch() {
  tf_executor.graph {
    %0:5 = tf_executor.island {
      %a = "tf.opA"() : () -> tensor<i1>
      // expected-error@+1 {{'tf_device.launch' op must contain only 'tf' dialect ops}}
      %launch:2 = "tf_device.launch"() ( {
        %b = "tf.opB"(%a) : (tensor<i1>) -> tensor<i32>
        %c = addi %b, %b : tensor<i32>
        tf_device.return %c, %b : tensor<i32>, tensor<i32>
      }) {device = "CPU:0"} : () -> (tensor<f32>, tensor<i32>)
      %d = "tf.opD"() : () -> tensor<i1>
      tf_executor.yield %a, %launch#0, %launch#1, %d : tensor<i1>, tensor<f32>, tensor<i32>, tensor<i1>
    }
    tf_executor.fetch
  }
  return
}


// -----


// Tests TensorFlow op with conflicting `device` attribute compared to parent
// `tf_device.launch`.
func @conflicting_device() {
  tf_executor.graph {
    %0 = tf_executor.island {
      // expected-error@+1 {{'tf_device.launch' op inner 'tf' dialect op has conflicting 'device' attribute, got 'GPU:0' but expected 'CPU:0'}}
      "tf_device.launch"() ( {
        "tf.opA"() {device = "GPU:0"} : () -> ()
        tf_device.return
      }) {device = "CPU:0"} : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}


// -----


// Tests TensorFlow op with bad `device` attribute already set.
func @bad_tf_device_attr() {
  tf_executor.graph {
    %0 = tf_executor.island {
      // expected-error@+1 {{'tf_device.launch' op inner 'tf' dialect op has bad 'device' attribute}}
      "tf_device.launch"() ( {
        "tf.opA"() {device = 0 : i32} : () -> ()
        tf_device.return
      }) {device = "CPU:0"} : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}
