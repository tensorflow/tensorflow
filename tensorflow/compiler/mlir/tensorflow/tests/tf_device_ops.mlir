// RUN: tf-opt %s | tf-opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @return_no_operands
func @return_no_operands() {
  "tf_device.launch"() ( {
// CHECK:   tf_device.return
    tf_device.return
  }) {device = "device"} : () -> ()
  return
}

// CHECK-LABEL: func @return_one_operand
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xf32>)
func @return_one_operand(%arg_0: tensor<*xf32>) {
  %result = "tf_device.launch"() ( {
// CHECK:   tf_device.return %[[ARG_0]] : tensor<*xf32>
    tf_device.return %arg_0 : tensor<*xf32>
  }) {device = "device"} : () -> tensor<*xf32>
  return
}

// CHECK-LABEL: func @return_multiple_operands
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<*xi32>)
func @return_multiple_operands(%arg_0: tensor<*xf32>, %arg_1: tensor<*xi32>) {
  %result:2 = "tf_device.launch"() ( {
// CHECK:   tf_device.return %[[ARG_0]], %[[ARG_1]] : tensor<*xf32>, tensor<*xi32>
    tf_device.return %arg_0, %arg_1 : tensor<*xf32>, tensor<*xi32>
  }) {device = "device"} : () -> (tensor<*xf32>, tensor<?xi32>)
  return
}

// CHECK-LABEL: func @empty_replicate
func @empty_replicate() {
  tf_device.replicate {n = 2 : i32} {
  }
  return

// CHECK:      tf_device.replicate
// CHECK-SAME: n = 2
// CHECK-NEXT:   tf_device.return
}

// CHECK-LABEL: func @replicate_with_multiple_operands
func @replicate_with_multiple_operands() {
  %0 = "tf.opA"() : () -> (tensor<*xi1>)
  %1 = "tf.opB"() : () -> (tensor<*xi1>)
  %2 = "tf.opC"() : () -> (tensor<*xi1>)
  %3 = "tf.opD"() : () -> (tensor<*xi32>)
  %4 = "tf.opE"() : () -> (tensor<*xi32>)
  %5 = "tf.opF"() : () -> (tensor<*xi32>)
  %6 = "tf.opG"() : () -> (tensor<*xf32>)
  %7 = "tf.opH"() : () -> (tensor<*xf32>)
  %8 = "tf.opI"() : () -> (tensor<*xf32>)
  tf_device.replicate([%0, %1, %2] as %input0: tensor<*xi1>, [%3, %4, %5] as %input1: tensor<*xi32>, [%6, %7, %8] as %input2: tensor<*xf32>) {n = 3 : i32} {
    tf_device.return
  }
  return

// CHECK:      %[[OP_A:[a-z0-9]*]] = "tf.opA"
// CHECK:      %[[OP_B:[a-z0-9]*]] = "tf.opB"
// CHECK:      %[[OP_C:[a-z0-9]*]] = "tf.opC"
// CHECK:      %[[OP_D:[a-z0-9]*]] = "tf.opD"
// CHECK:      %[[OP_E:[a-z0-9]*]] = "tf.opE"
// CHECK:      %[[OP_F:[a-z0-9]*]] = "tf.opF"
// CHECK:      %[[OP_G:[a-z0-9]*]] = "tf.opG"
// CHECK:      %[[OP_H:[a-z0-9]*]] = "tf.opH"
// CHECK:      %[[OP_I:[a-z0-9]*]] = "tf.opI"
// CHECK:      tf_device.replicate
// CHECK-SAME: ([%[[OP_A]], %[[OP_B]], %[[OP_C]]] as %{{[a-z0-9]*}}: tensor<*xi1>, [%[[OP_D]], %[[OP_E]], %[[OP_F]]] as %{{[a-z0-9]*}}: tensor<*xi32>, [%[[OP_G]], %[[OP_H]], %[[OP_I]]] as %{{[a-z0-9]*}}: tensor<*xf32>)
// CHECK-SAME: n = 3
// CHECK-NEXT:   tf_device.return
}

// CHECK-LABEL: func @replicate_with_return
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<*xf32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xi32>)
func @replicate_with_return(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xi32>) {
  %result:4 = tf_device.replicate([%arg0, %arg1] as %input0: tensor<*xf32>) {n = 2 : i32} {
    tf_device.return %input0, %arg2 : tensor<*xf32>, tensor<*xi32>
  }
  return

// CHECK:      tf_device.replicate
// CHECK-SAME: ([%[[ARG_0]], %[[ARG_1]]] as %[[INPUT_0:[a-z0-9]*]]: tensor<*xf32>)
// CHECK-SAME: n = 2
// CHECK-NEXT:   tf_device.return %[[INPUT_0]], %[[ARG_2]]
}

// CHECK-LABEL: func @replicate_with_devices
func @replicate_with_devices() {
  tf_device.replicate() {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/DEVICE:0", "/DEVICE:1"]}} {
    tf_device.return
  }
  return

// CHECK:      tf_device.replicate
// CHECK-SAME: devices = {TPU_REPLICATED_CORE_0 = ["/DEVICE:0", "/DEVICE:1"]}
// CHECK-SAME: n = 2
// CHECK-NEXT:   tf_device.return
}

// CHECK-LABEL: func @replicate_with_multiple_devices
func @replicate_with_multiple_devices() {
  tf_device.replicate() {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/DEVICE:0", "/DEVICE:1"], TPU_REPLICATED_CORE_1 = ["/DEVICE:2", "/DEVICE:3"]}} {
    tf_device.return
  }
  return

// CHECK:      tf_device.replicate
// CHECK-SAME: devices = {TPU_REPLICATED_CORE_0 = ["/DEVICE:0", "/DEVICE:1"], TPU_REPLICATED_CORE_1 = ["/DEVICE:2", "/DEVICE:3"]}
// CHECK-SAME: n = 2
// CHECK-NEXT:   tf_device.return
}

// CHECK-LABEL: func @replicate_with_inner_ops
func @replicate_with_inner_ops() {
  %0 = "tf.opA"() : () -> (tensor<*xi1>)
  %1 = "tf.opB"() : () -> (tensor<*xi1>)
  %2 = "tf.opC"() : () -> (tensor<*xi32>)
  %3 = "tf.opD"() : () -> (tensor<*xi32>)
  %4 = "tf.opE"() : () -> (tensor<*xf32>)
  %result:4 = tf_device.replicate([%0, %1] as %input0: tensor<*xi1>, [%2, %3] as %input1: tensor<*xi32>) {n = 2 : i32} {
    %5 = "tf.opF"(%input0, %4) : (tensor<*xi1>, tensor<*xf32>) -> (tensor<*xi1>)
    %6 = "tf.opG"(%input1, %4) : (tensor<*xi32>, tensor<*xf32>) -> (tensor<*xi32>)
    tf_device.return %5, %6 : tensor<*xi1>, tensor<*xi32>
  }
  return
}

// CHECK-LABEL: func @parallel_execute_two_regions
func @parallel_execute_two_regions() {
  "tf_device.parallel_execute"() ({
    tf_device.return
  },
  {
    tf_device.return
  }) {} : () -> ()
  return
}

// CHECK-LABEL: func @parallel_execute_two_regions_with_ops
func @parallel_execute_two_regions_with_ops() {
  "tf_device.parallel_execute"() ({
    %0 = "tf.opA"() : () -> (tensor<*xi1>)
    %1 = "tf.opB"() : () -> (tensor<*xi32>)
    tf_device.return %0, %1 : tensor<*xi1>, tensor<*xi32>
  },
  {
    %2 = "tf.opC"() : () -> (tensor<*xi1>)
    tf_device.return
  }) {} : () -> (tensor<*xi1>, tensor<*xi32>)
  return
}

// CHECK-LABEL: func @parallel_execute_regions_with_data_results
func @parallel_execute_regions_with_data_results() {
  "tf_device.parallel_execute"() ({
    %0 = "tf.opA"() : () -> (tensor<*xi1>)
    %1 = "tf.opB"() : () -> (tensor<*xi32>)
    tf_device.return %0, %1 : tensor<*xi1>, tensor<*xi32>
  },
  {
    %2 = "tf.opC"() : () -> (tensor<*xf32>)
    tf_device.return %2 : tensor<*xf32>
  }) {} : () -> (tensor<*xi1>, tensor<*xi32>, tensor<*xf32>)
  return
}
