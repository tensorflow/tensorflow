// RUN: tf-opt %s -tf-tpu-device-propagation | FileCheck %s

// Tests function passthrough values.

// CHECK-LABEL: func @testArgToRet
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
// CHECK-SAME: -> (tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
func @testArgToRet(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> tensor<i64> {
  %0 = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<i64>
  }
  return %0 : tensor<i64>
}

// Tests supported ops.

// CHECK-LABEL: func @testIdentityOp
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
// CHECK-SAME: -> (tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
func @testIdentityOp(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> tensor<i64> {
  %0 = tf_executor.graph {
    // CHECK:      tf.Identity
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.island wraps "tf.Identity"(%arg0) : (tensor<i64>) -> tensor<i64>
    tf_executor.fetch %1#0 : tensor<i64>
  }
  return %0 : tensor<i64>
}

// CHECK-LABEL: func @testIdentityNOp
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, {{%.+}}: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
// CHECK-SAME: -> (tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
func @testIdentityNOp(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg1: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> (tensor<i64>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    // CHECK:      tf.IdentityN
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:3 = tf_executor.island wraps "tf.IdentityN"(%arg0, %arg1) : (tensor<i64>, tensor<i32>) -> (tensor<i64>, tensor<i32>)
    tf_executor.fetch %1#0, %1#1 : tensor<i64>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i64>, tensor<i32>
}

// CHECK-LABEL: func @testShapeOp
// CHECK-SAME: ({{%.+}}: tensor<*xi64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
// CHECK-SAME: -> (tensor<?xi64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
func @testShapeOp(%arg0: tensor<*xi64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> tensor<?xi64> {
  %0 = tf_executor.graph {
    // CHECK:      tf.Shape
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.island wraps "tf.Shape"(%arg0) : (tensor<*xi64>) -> tensor<?xi64>
    tf_executor.fetch %1#0 : tensor<?xi64>
  }
  return %0 : tensor<?xi64>
}

// CHECK-LABEL: func @testEnterOp
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
// CHECK-SAME: -> (tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
func @testEnterOp(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> tensor<i64> {
  %0 = tf_executor.graph {
    // CHECK:      tf_executor.Enter
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.Enter %arg0 frame "frame" : tensor<i64>
    tf_executor.fetch %1#0 : tensor<i64>
  }
  return %0 : tensor<i64>
}

// CHECK-LABEL: func @testExitOp
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
// CHECK-SAME: -> (tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
func @testExitOp(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> tensor<i64> {
  %0 = tf_executor.graph {
    // CHECK:      tf_executor.Exit
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.Exit %arg0 : tensor<i64>
    tf_executor.fetch %1#0 : tensor<i64>
  }
  return %0 : tensor<i64>
}

// CHECK-LABEL: func @testMergeOp
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, {{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
// CHECK-SAME: -> (tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
func @testMergeOp(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg1: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> (tensor<i64>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    // CHECK:      tf_executor.Merge
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:3 = tf_executor.Merge %arg0, %arg1 : tensor<i64>
    tf_executor.fetch %1#0, %1#1 : tensor<i64>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i64>, tensor<i32>
}

// CHECK-LABEL: func @testSwitchOp
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, {{%.+}}: tensor<i1> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
func @testSwitchOp(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg1: tensor<i1> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) {
  tf_executor.graph {
    // CHECK:      tf_executor.Switch
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %0:3 = tf_executor.Switch %arg0, %arg1 : tensor<i64> {T = "tfdtype$DT_INT64"}
    // CHECK:      tf.Identity
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<i64>) -> tensor<i64>
    // CHECK:      tf.Identity
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %2:2 = tf_executor.island wraps "tf.Identity"(%0#1) : (tensor<i64>) -> tensor<i64>
    %3 = tf_executor.ControlTrigger %1#1, %2#1
    tf_executor.fetch %3 : !tf_executor.control
  }
  return
}

// Tests unsupported op does not have TPU device propagated.

// CHECK-LABEL: func @testUnsupportedOp
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
// CHECK-SAME: -> tensor<i64>
func @testUnsupportedOp(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> tensor<i64> {
  %0 = tf_executor.graph {
    // CHECK:      tf.UnsupportedOp
    // CHECK-NOT:  device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.island wraps "tf.UnsupportedOp"(%arg0) : (tensor<i64>) -> tensor<i64>
    tf_executor.fetch %1#0 : tensor<i64>
  }
  return %0 : tensor<i64>
}

// Tests empty devices are overwritten.

// CHECK-LABEL: func @testEmptyDeviceOverwritten
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
// CHECK-SAME: -> (tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
func @testEmptyDeviceOverwritten(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> (tensor<i64> {tf.device = ""}) {
  %0 = tf_executor.graph {
    // CHECK:      tf.Identity
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.island wraps "tf.Identity"(%arg0) {device = ""} : (tensor<i64>) -> tensor<i64>
    tf_executor.fetch %1#0 : tensor<i64>
  }
  return %0 : tensor<i64>
}

// Tests only devices are propagated when all operands are on the same TPU
// device.

// CHECK-LABEL: func @testOperandsNoDevice
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, {{%.+}}: tensor<i32>)
// CHECK-SAME: -> (tensor<i64>, tensor<i32>)
func @testOperandsNoDevice(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg1: tensor<i32>) -> (tensor<i64>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    // CHECK:      tf.IdentityN
    // CHECK-NOT:  device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:3 = tf_executor.island wraps "tf.IdentityN"(%arg0, %arg1) : (tensor<i64>, tensor<i32>) -> (tensor<i64>, tensor<i32>)
    tf_executor.fetch %1#0, %1#1 : tensor<i64>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i64>, tensor<i32>
}

// CHECK-LABEL: func @testOperandsDifferentDevice
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, {{%.+}}: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:1"})
// CHECK-SAME: -> (tensor<i64>, tensor<i32>)
func @testOperandsDifferentDevice(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg1: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:1"}) -> (tensor<i64>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    // CHECK:      tf.IdentityN
    // CHECK-NOT:  device = "/job:localhost/replica:0/task:0/device:TPU:0"
    // CHECK-NOT:  device = "/job:localhost/replica:0/task:0/device:TPU:1"
    %1:3 = tf_executor.island wraps "tf.IdentityN"(%arg0, %arg1) : (tensor<i64>, tensor<i32>) -> (tensor<i64>, tensor<i32>)
    tf_executor.fetch %1#0, %1#1 : tensor<i64>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i64>, tensor<i32>
}

// Tests op with operand on different device does not have its device
// overwritten.

// CHECK-LABEL: func @testDifferentOperandAndOpDevice
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
func @testDifferentOperandAndOpDevice(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) {
  tf_executor.graph {
    // CHECK:      tf.Identity
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:1"
    %0:2 = tf_executor.island wraps "tf.Identity"(%arg0) {device = "/job:localhost/replica:0/task:0/device:TPU:1"} : (tensor<i64>) -> tensor<i64>
    tf_executor.fetch %0#1 : !tf_executor.control
  }
  return
}

// CHECK-LABEL: func @testDifferentOperandAndResultDevice
// CHECK-SAME: ({{%.+}}: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"})
// CHECK-SAME: -> (tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:1"})
func @testDifferentOperandAndResultDevice(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> (tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:1"}) {
  %0 = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<i64>
  }
  return %0 : tensor<i64>
}

// Tests non TPU devices are not propagated.

// CHECK-LABEL: func @testNonTPUDevice
func @testNonTPUDevice(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) {
  tf_executor.graph {
    // CHECK:      tf.Identity
    // CHECK-NOT:  device = "/job:localhost/replica:0/task:0/device:CPU:0"
    %0:2 = tf_executor.island wraps "tf.Identity"(%arg0) : (tensor<i64>) -> tensor<i64>
    tf_executor.fetch %0#1 : !tf_executor.control
  }
  return
}

// Tests control dependencies are ignored for propagating devices.

// CHECK-LABEL: func @testControlDependenciesIgnored
func @testControlDependenciesIgnored(%arg0: tensor<i64>) {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:TPU:0", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    // CHECK:      tf.Identity
    // CHECK-NOT:  device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.island(%0#1) wraps "tf.Identity"(%arg0) : (tensor<i64>) -> tensor<i64>
    tf_executor.fetch %1#1 : !tf_executor.control
  }
  return
}

// CHECK-LABEL: func @testControlDependenciesMismatchedDevices
func @testControlDependenciesMismatchedDevices(%arg0: tensor<i64> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:TPU:1", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    // CHECK:      tf.Identity
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.island(%0#1) wraps "tf.Identity"(%arg0) : (tensor<i64>) -> tensor<i64>
    tf_executor.fetch %1#1 : !tf_executor.control
  }
  return
}

// Tests LoopCond -> Switch where LoopCond has a different device is ignored.

// CHECK-LABEL: func @testLoopCondSwitchLinkDifferentDevice
func @testLoopCondSwitchLinkDifferentDevice() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %1:2 = tf_executor.LoopCond %0#0 : (tensor<i1>) -> (tensor<i1>, !tf_executor.control) {}
    %2:2 = tf_executor.island wraps "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:TPU:0", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    // CHECK:      tf_executor.Switch
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %3:3 = tf_executor.Switch %2#0, %1#0 : tensor<i64> {T = "tfdtype$DT_INT64"}
    // CHECK:      tf.Identity
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %4:2 = tf_executor.island wraps "tf.Identity"(%3#0) : (tensor<i64>) -> tensor<i64>
    // CHECK:      tf.Identity
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %5:2 = tf_executor.island wraps "tf.Identity"(%3#1) : (tensor<i64>) -> tensor<i64>
    %6 = tf_executor.ControlTrigger %4#1, %5#1
    tf_executor.fetch %6 : !tf_executor.control
  }
  return
}

// Tests tf_executor.NextIteration.Source/tf_executor.NextIteration.Sink has a
// device when an intermediate op in its loop has a device.

// CHECK-LABEL: func @testNextIterationNoDevice
func @testNextIterationNoDevice() {
  tf_executor.graph {
    // CHECK:      tf_executor.NextIteration.Source
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %0:3 = tf_executor.NextIteration.Source : tensor<i64> {T = "tfdtype$DT_INT64"}
    // CHECK:      tf.Identity
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<i64>) -> tensor<i64>
    // CHECK:      tf.IdentityN
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %2:2 = tf_executor.island wraps "tf.IdentityN"(%1#0) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : (tensor<i64>) -> tensor<i64>
    // CHECK:      tf_executor.NextIteration.Sink
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    tf_executor.NextIteration.Sink [%0#1] %2#0 : tensor<i64> {T = "tfdtype$DT_INT64"}
    tf_executor.fetch %0#2 : !tf_executor.control
  }
  return
}

// Tests tf_executor.NextIteration with mismatched devices does not propagate
// either device.

// CHECK-LABEL: func @testNextIterationMismatchedDevices
func @testNextIterationMismatchedDevices() {
  tf_executor.graph {
    // CHECK:      tf_executor.NextIteration.Source
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:1"
    %0:3 = tf_executor.NextIteration.Source : tensor<i64> {device = "/job:localhost/replica:0/task:0/device:TPU:1", T = "tfdtype$DT_INT64"}
    // CHECK:      "tf.Identity"({{.+}}) :
    %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<i64>) -> tensor<i64>
    // CHECK:      tf_executor.NextIteration.Sink
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    tf_executor.NextIteration.Sink [%0#1] %1#0 : tensor<i64> {device = "/job:localhost/replica:0/task:0/device:TPU:0", T = "tfdtype$DT_INT64"}
    tf_executor.fetch %0#2 : !tf_executor.control
  }
  return
}

// CHECK-LABEL: func @testNextIterationMissingSourceDevice
func @testNextIterationMissingSourceDevice() {
  tf_executor.graph {
    // CHECK:      tf_executor.NextIteration.Source
    %0:3 = tf_executor.NextIteration.Source : tensor<i64> {T = "tfdtype$DT_INT64"}
    // CHECK:      "tf.Identity"({{.+}}) :
    %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<i64>) -> tensor<i64>
    // CHECK:      tf_executor.NextIteration.Sink
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    tf_executor.NextIteration.Sink [%0#1] %1#0 : tensor<i64> {device = "/job:localhost/replica:0/task:0/device:TPU:0", T = "tfdtype$DT_INT64"}
    tf_executor.fetch %0#2 : !tf_executor.control
  }
  return
}

// CHECK-LABEL: func @testNextIterationMissingSinkDevice
func @testNextIterationMissingSinkDevice() {
  tf_executor.graph {
    // CHECK:      tf_executor.NextIteration.Source
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:1"
    %0:3 = tf_executor.NextIteration.Source : tensor<i64> {device = "/job:localhost/replica:0/task:0/device:TPU:1", T = "tfdtype$DT_INT64"}
    // CHECK:      "tf.Identity"({{.+}}) :
    %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<i64>) -> tensor<i64>
    // CHECK:      tf_executor.NextIteration.Sink
    tf_executor.NextIteration.Sink [%0#1] %1#0 : tensor<i64> {T = "tfdtype$DT_INT64"}
    tf_executor.fetch %0#2 : !tf_executor.control
  }
  return
}

// Tests unsupported functions are not modified.

// CHECK-LABEL: func @testMultipleBlockFunc
func @testMultipleBlockFunc() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:TPU:0", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    // CHECK:      tf.Identity
    // CHECK-NOT:  device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<i64>) -> tensor<i64>
    tf_executor.fetch %1#1 : !tf_executor.control
  }
  br ^bb1
^bb1:
  return
}

// CHECK-LABEL: func @testMultipleGraphs
func @testMultipleGraphs() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:TPU:0", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    // CHECK:      tf.Identity
    // CHECK-NOT:  device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<i64>) -> tensor<i64>
    tf_executor.fetch %1#1 : !tf_executor.control
  }
  tf_executor.graph {
    tf_executor.fetch
  }
  return
}

// CHECK-LABEL: func @testNoGraph
func @testNoGraph() -> tensor<i64> {
  %0 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:TPU:0", value = dense<0> : tensor<i64>} : () -> tensor<i64>
  // CHECK:      tf.Identity
  // CHECK-NOT:  device = "/job:localhost/replica:0/task:0/device:TPU:0"
  %1 = "tf.Identity"(%0) : (tensor<i64>) -> tensor<i64>
  return %1 : tensor<i64>
}

// CHECK-LABEL: func @testMismatchedGraphResults
func @testMismatchedGraphResults() {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:TPU:0", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    // CHECK:      tf.Identity
    // CHECK-NOT:  device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %2:2 = tf_executor.island wraps "tf.Identity"(%1#0) : (tensor<i64>) -> tensor<i64>
    tf_executor.fetch %2#0 : tensor<i64>
  }
  return
}
