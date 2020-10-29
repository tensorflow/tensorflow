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


// Tests empty device string attributes are overwritten.
// CHECK-LABEL: func @empty_device_op
func @empty_device_op() {
  tf_executor.graph {
    %0:3 = tf_executor.island {
      %launch:2 = "tf_device.launch"() ( {
        %a:2 = "tf.opA"() {device = ""} : () -> (tensor<i32>, tensor<f32>)
        tf_device.return %a#1, %a#0 : tensor<f32>, tensor<i32>
      }) {device = "CPU:0"} : () -> (tensor<f32>, tensor<i32>)
      tf_executor.yield %launch#0, %launch#1: tensor<f32>, tensor<i32>
    }
    tf_executor.fetch
  }
  return
}

// CHECK:      [[A:%.+]]:2 = "tf.opA"
// CHECK-SAME: device = "CPU:0"
// CHECK-NOT:  tf_device.launch
// CHECK:      tf_executor.yield [[A]]#1, [[A]]#0


// Tests devices are propagated into tf.Case op branches.
// CHECK-LABEL: func @case
func @case(%arg0: tensor<i32>) {
  tf_executor.graph {
    %0 = tf_executor.island {
      "tf_device.launch"() ( {
        "tf.Case"(%arg0) {branches = [@case_branch], is_stateless = false} : (tensor<i32>) -> ()
        tf_device.return
      }) {device = "CPU:0"} : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch %0 : !tf_executor.control
  }
  return
}

// CHECK:      tf.Case
// CHECK-SAME: device = "CPU:0"

// CHECK-LABEL: func @case_branch
func @case_branch() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.opA"() : () -> tensor<i1>
    tf_executor.fetch %0#1 : !tf_executor.control
  }
  return
}

// CHECK:      tf.opA
// CHECK-SAME: device = "CPU:0"


// Tests devices are propagated into tf.If op branches.
// CHECK-LABEL: func @if
func @if(%arg0: tensor<i1>) {
  tf_executor.graph {
    %0 = tf_executor.island {
      "tf_device.launch"() ( {
        "tf.If"(%arg0) {then_branch = @then_branch, else_branch = @else_branch, is_stateless = false} : (tensor<i1>) -> ()
        tf_device.return
      }) {device = "CPU:0"} : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch %0 : !tf_executor.control
  }
  return
}

// CHECK:      tf.If
// CHECK-SAME: device = "CPU:0"

// CHECK-LABEL: func @then_branch
func @then_branch() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.opB"() : () -> tensor<i1>
    tf_executor.fetch %0#1 : !tf_executor.control
  }
  return
}

// CHECK:      tf.opB
// CHECK-SAME: device = "CPU:0"

// CHECK-LABEL: func @else_branch
func @else_branch() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.opC"() : () -> tensor<i1>
    tf_executor.fetch %0#1 : !tf_executor.control
  }
  return
}

// CHECK:      tf.opC
// CHECK-SAME: device = "CPU:0"


// Tests devices are propagated into tf.While op functions.
// CHECK-LABEL: func @while
func @while(%arg0: tensor<i1>) {
  tf_executor.graph {
    %0 = tf_executor.island {
      "tf_device.launch"() ( {
        %1 = "tf.While"(%arg0) {cond = @cond_func, body = @body_func, is_stateless = false} : (tensor<i1>) -> tensor<i1>
        tf_device.return
      }) {device = "CPU:0"} : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch %0 : !tf_executor.control
  }
  return
}

// CHECK:      tf.While
// CHECK-SAME: device = "CPU:0"

// CHECK-LABEL: func @cond_func
func @cond_func(%arg0: tensor<i1>) -> tensor<i1> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.opD"(%arg0) : (tensor<i1>) -> tensor<i1>
    tf_executor.fetch %1#0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

// CHECK:      tf.opD
// CHECK-SAME: device = "CPU:0"

// CHECK-LABEL: func @body_func
func @body_func(%arg0: tensor<i1>) -> tensor<i1> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.opE"(%arg0) : (tensor<i1>) -> tensor<i1>
    tf_executor.fetch %1#0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

// CHECK:      tf.opE
// CHECK-SAME: device = "CPU:0"


// Tests devices are propagated into functions.
// CHECK-LABEL: func @call(
func @call() {
  tf_executor.graph {
    %0 = tf_executor.island {
      "tf_device.launch"() ( {
        "tf.StatefulPartitionedCall"() {f = @call_func0, config = "", config_proto = "", executor_type = ""} : () -> ()
        tf_device.return
      }) {device = "CPU:0"} : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch %0 : !tf_executor.control
  }
  return
}

// CHECK:      tf.StatefulPartitionedCall
// CHECK-SAME: device = "CPU:0"

// CHECK-LABEL: func @call_func0
func @call_func0() {
  tf_executor.graph {
    %0 = tf_executor.island wraps "tf.StatefulPartitionedCall"() {f = @call_func1, config = "", config_proto = "", executor_type = ""} : () -> ()
    tf_executor.fetch %0 : !tf_executor.control
  }
  return
}

// CHECK:      tf.StatefulPartitionedCall
// CHECK-SAME: device = "CPU:0"

// CHECK-LABEL: func @call_func1
func @call_func1() {
  tf_executor.graph {
    %0 = tf_executor.island wraps "tf.opF"() : () -> ()
    tf_executor.fetch %0 : !tf_executor.control
  }
  return
}

// CHECK:      tf.opF
// CHECK-SAME: device = "CPU:0"


// Test v1 control flow ops reachable from a tf_device.launch have devices
// assigned.
// CHECK-LABEL: func @call_to_graph
func @call_to_graph() {
  tf_executor.graph {
    %0 = tf_executor.island {
      "tf_device.launch"() ( {
        "tf.StatefulPartitionedCall"() {f = @v1_control_flow, config = "", config_proto = "", executor_type = ""} : () -> ()
        tf_device.return
      }) {device = "CPU:0"} : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch %0 : !tf_executor.control
  }
  return
}

// CHECK:      tf.StatefulPartitionedCall
// CHECK-SAME: device = "CPU:0"

// CHECK-LABEL: func @v1_control_flow
func @v1_control_flow() {
  %0 = tf_executor.graph {
    %0:3 = tf_executor.NextIteration.Source : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    %1:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %2:2 = tf_executor.Enter %1#0 frame "while/while_context" : (tensor<i32>) -> (tensor<*xi32>, !tf_executor.control) {T = "tfdtype$DT_INT32"}
    %3:3 = tf_executor.Merge %2#0, %0#0 : tensor<*xi32> {N = 2 : i64, T = "tfdtype$DT_INT32"}
    %4:2 = tf_executor.island(%3#2) wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<10> : tensor<i32>} : () -> tensor<i32>
    %5:2 = tf_executor.island wraps "tf.Less"(%3#0, %4#0) {T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
    %6:2 = tf_executor.LoopCond %5#0 : (tensor<*xi1>) -> (tensor<i1>, !tf_executor.control) {}
    %7:3 = tf_executor.Switch %3#0, %6#0 : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    %8:2 = tf_executor.Exit %7#0 : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    %9:2 = tf_executor.island wraps "tf.Identity"(%7#1) {T = "tfdtype$DT_INT32"} : (tensor<*xi32>) -> tensor<*xi32>
    %10:2 = tf_executor.island(%9#1) wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %11:2 = tf_executor.island wraps "tf.Add"(%9#0, %10#0) {T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    tf_executor.NextIteration.Sink [%0#1] %11#0 : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    tf_executor.fetch %8#0 : tensor<*xi32>
  }
  return
}

// CHECK:      tf_executor.NextIteration.Source
// CHECK-SAME: device = "CPU:0"
// CHECK:      tf_executor.Enter
// CHECK-SAME: device = "CPU:0"
// CHECK:      tf_executor.Merge
// CHECK-SAME: device = "CPU:0"
// CHECK:      tf_executor.LoopCond
// CHECK-SAME: device = "CPU:0"
// CHECK:      tf_executor.Switch
// CHECK-SAME: device = "CPU:0"
// CHECK:      tf_executor.Exit
// CHECK-SAME: device = "CPU:0"
// CHECK:      tf_executor.NextIteration.Sink
// CHECK-SAME: device = "CPU:0"


// -----


// Tests TensorFlow op with conflicting `device` attribute compared to parent
// `tf_device.launch`.
func @conflicting_device() {
  tf_executor.graph {
    %0 = tf_executor.island {
      // expected-error@+1 {{'tf_device.launch' op inner op has conflicting 'device' attribute, got 'GPU:0' but expected 'CPU:0'}}
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


// Tests TensorFlow op with conflicting `device` attribute compared to parent
// `tf_device.launch` via a reachable function.
func @conflicting_device_function() {
  tf_executor.graph {
    %0 = tf_executor.island {
      // expected-error@+1 {{'tf_device.launch' op inner op has conflicting 'device' attribute, got 'GPU:0' but expected 'CPU:0'}}
      "tf_device.launch"() ( {
        "tf.StatefulPartitionedCall"() {f = @callee, config = "", config_proto = "", executor_type = ""} : () -> ()
        tf_device.return
      }) {device = "CPU:0"} : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch %0 : !tf_executor.control
  }
  return
}

func @callee() {
  tf_executor.graph {
    %0 = tf_executor.island wraps "tf.opA"() {device = "GPU:0"} : () -> ()
    tf_executor.fetch %0 : !tf_executor.control
  }
  return
}


// -----


// Tests TensorFlow op with bad `device` attribute already set.
func @bad_tf_device_attr() {
  tf_executor.graph {
    %0 = tf_executor.island {
      // expected-error@+1 {{'tf_device.launch' op inner op has bad 'device' attribute}}
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
