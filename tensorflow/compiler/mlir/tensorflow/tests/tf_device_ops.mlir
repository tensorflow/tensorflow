// RUN: tf-opt --verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @return_no_operands
func.func @return_no_operands() {
  "tf_device.launch"() ({
// CHECK:   tf_device.return
    tf_device.return
  }) {device = "device"} : () -> ()
  func.return
}

// -----

// CHECK-LABEL: func @return_one_operand
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xf32>)
func.func @return_one_operand(%arg_0: tensor<*xf32>) {
  %result = "tf_device.launch"() ({
// CHECK:   tf_device.return %[[ARG_0]] : tensor<*xf32>
    tf_device.return %arg_0 : tensor<*xf32>
  }) {device = "device"} : () -> tensor<*xf32>
  func.return
}

// -----

// CHECK-LABEL: func @return_multiple_operands
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<*xi32>)
func.func @return_multiple_operands(%arg_0: tensor<*xf32>, %arg_1: tensor<*xi32>) {
  %result:2 = "tf_device.launch"() ({
// CHECK:   tf_device.return %[[ARG_0]], %[[ARG_1]] : tensor<*xf32>, tensor<*xi32>
    tf_device.return %arg_0, %arg_1 : tensor<*xf32>, tensor<*xi32>
  }) {device = "device"} : () -> (tensor<*xf32>, tensor<?xi32>)
  func.return
}

// -----

// CHECK-LABEL: func @empty_replicate
func.func @empty_replicate() {
  tf_device.replicate {n = 2 : i32} {
  }
  func.return

// CHECK:      tf_device.replicate
// CHECK-SAME: n = 2
// CHECK-NEXT:   tf_device.return
}

// -----

// CHECK-LABEL: func @no_operand_replicate
func.func @no_operand_replicate() {
  tf_device.replicate {n = 2 : i32} {
    %0 = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
    %1 = "tf.Const"() { value = dense<1> : tensor<i64> } : () -> tensor<i64>
    tf_device.return %0, %1 : tensor<i64>, tensor<i64>
  }
  func.return
  // CHECK:      tf_device.replicate
  // CHECK-SAME: n = 2
  // CHECK:   tf_device.return
}

// -----

// CHECK-LABEL: func @replicate_with_multiple_operands
func.func @replicate_with_multiple_operands() {
  %0 = "tf.opA"() : () -> tensor<*xi1>
  %1 = "tf.opB"() : () -> tensor<*xi1>
  %2 = "tf.opC"() : () -> tensor<*xi1>
  %3 = "tf.opD"() : () -> tensor<*xi32>
  %4 = "tf.opE"() : () -> tensor<*xi32>
  %5 = "tf.opF"() : () -> tensor<*xi32>
  %6 = "tf.opG"() : () -> tensor<*xf32>
  %7 = "tf.opH"() : () -> tensor<*xf32>
  %8 = "tf.opI"() : () -> tensor<*xf32>
  %9 = "tf.opJ"() : () -> tensor<*xi8>
  %10 = "tf.opK"() : () -> tensor<*xi16>
  %11 = "tf.opL"() : () -> tensor<*xi64>
  tf_device.replicate([%0, %1, %2] as %input0: tensor<*xi1>, %9 as %input1: tensor<*xi8>, %10 as %input2: tensor<*xi16>, [%3, %4, %5] as %input3: tensor<*xi32>, [%6, %7, %8] as %input4: tensor<*xf32>, %11 as %input5: tensor<*xi64>) {n = 3 : i32} {
    tf_device.return
  }
  func.return

// CHECK:      %[[OP_A:[a-z0-9]*]] = "tf.opA"
// CHECK:      %[[OP_B:[a-z0-9]*]] = "tf.opB"
// CHECK:      %[[OP_C:[a-z0-9]*]] = "tf.opC"
// CHECK:      %[[OP_D:[a-z0-9]*]] = "tf.opD"
// CHECK:      %[[OP_E:[a-z0-9]*]] = "tf.opE"
// CHECK:      %[[OP_F:[a-z0-9]*]] = "tf.opF"
// CHECK:      %[[OP_G:[a-z0-9]*]] = "tf.opG"
// CHECK:      %[[OP_H:[a-z0-9]*]] = "tf.opH"
// CHECK:      %[[OP_I:[a-z0-9]*]] = "tf.opI"
// CHECK:      %[[OP_J:[a-z0-9]*]] = "tf.opJ"
// CHECK:      %[[OP_K:[a-z0-9]*]] = "tf.opK"
// CHECK:      %[[OP_L:[a-z0-9]*]] = "tf.opL"
// CHECK:      tf_device.replicate
// CHECK-SAME: [%[[OP_A]], %[[OP_B]], %[[OP_C]]] as %{{[a-z0-9]*}}: tensor<*xi1>
// CHECK-SAME: [%[[OP_D]], %[[OP_E]], %[[OP_F]]] as %{{[a-z0-9]*}}: tensor<*xi32>
// CHECK-SAME: [%[[OP_G]], %[[OP_H]], %[[OP_I]]] as %{{[a-z0-9]*}}: tensor<*xf32>
// CHECK-SAME: %[[OP_J]] as %{{[a-z0-9]*}}: tensor<*xi8>
// CHECK-SAME: %[[OP_K]] as %{{[a-z0-9]*}}: tensor<*xi16>
// CHECK-SAME: %[[OP_L]] as %{{[a-z0-9]*}}: tensor<*xi64>
// CHECK-SAME: n = 3
// CHECK-NEXT:   tf_device.return
}

// -----

// CHECK-LABEL: func @replicate_derived_operandSegmentSizes
func.func @replicate_derived_operandSegmentSizes() {
  tf_device.replicate {n = 2 : i32, operandSegmentSizes = array<i32: 0, 0>} {
  }
  func.return

// CHECK:      tf_device.replicate
// CHECK-SAME: n = 2
// CHECK-NOT:  operandSegmentSizes
// CHECK-NEXT:   tf_device.return
}

// -----

// CHECK-LABEL: func @replicate_with_return
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<*xf32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xi32>)
func.func @replicate_with_return(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xi32>) {
  %result:4 = tf_device.replicate([%arg0, %arg1] as %input0: tensor<*xf32>) {n = 2 : i32} {
    tf_device.return %input0, %arg2 : tensor<*xf32>, tensor<*xi32>
  }
  func.return

// CHECK:      tf_device.replicate
// CHECK-SAME: ([%[[ARG_0]], %[[ARG_1]]] as %[[INPUT_0:[a-z0-9]*]]: tensor<*xf32>)
// CHECK-SAME: n = 2
// CHECK-NEXT:   tf_device.return %[[INPUT_0]], %[[ARG_2]]
}

// -----

// CHECK-LABEL: func @replicate_with_devices
func.func @replicate_with_devices() {
  tf_device.replicate() {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/DEVICE:0", "/DEVICE:1"]}} {
    tf_device.return
  }
  func.return

// CHECK:      tf_device.replicate
// CHECK-SAME: devices = {TPU_REPLICATED_CORE_0 = ["/DEVICE:0", "/DEVICE:1"]}
// CHECK-SAME: n = 2
// CHECK-NEXT:   tf_device.return
}

// -----

// CHECK-LABEL: func @replicate_with_multiple_devices
func.func @replicate_with_multiple_devices() {
  tf_device.replicate() {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/DEVICE:0", "/DEVICE:1"], TPU_REPLICATED_CORE_1 = ["/DEVICE:2", "/DEVICE:3"]}} {
    tf_device.return
  }
  func.return

// CHECK:      tf_device.replicate
// CHECK-SAME: devices = {TPU_REPLICATED_CORE_0 = ["/DEVICE:0", "/DEVICE:1"], TPU_REPLICATED_CORE_1 = ["/DEVICE:2", "/DEVICE:3"]}
// CHECK-SAME: n = 2
// CHECK-NEXT:   tf_device.return
}

// -----

// CHECK-LABEL: func @replicate_with_inner_ops
func.func @replicate_with_inner_ops() {
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
  func.return
}

// -----

// CHECK-LABEL: func @parallel_execute_two_regions
func.func @parallel_execute_two_regions() {
  "tf_device.parallel_execute"() ({
    tf_device.return
  },
  {
    tf_device.return
  }) {} : () -> ()
  func.return
}

// -----

// CHECK-LABEL: func @parallel_execute_two_regions_with_ops
func.func @parallel_execute_two_regions_with_ops() {
  "tf_device.parallel_execute"() ({
    %0 = "tf.opA"() : () -> (tensor<*xi1>)
    %1 = "tf.opB"() : () -> (tensor<*xi32>)
    tf_device.return %0, %1 : tensor<*xi1>, tensor<*xi32>
  },
  {
    %2 = "tf.opC"() : () -> (tensor<*xi1>)
    tf_device.return
  }) {} : () -> (tensor<*xi1>, tensor<*xi32>)
  func.return
}

// -----

// CHECK-LABEL: func @parallel_execute_regions_with_data_results
func.func @parallel_execute_regions_with_data_results() {
  "tf_device.parallel_execute"() ({
    %0 = "tf.opA"() : () -> (tensor<*xi1>)
    %1 = "tf.opB"() : () -> (tensor<*xi32>)
    tf_device.return %0, %1 : tensor<*xi1>, tensor<*xi32>
  },
  {
    %2 = "tf.opC"() : () -> (tensor<*xf32>)
    tf_device.return %2 : tensor<*xf32>
  }) {} : () -> (tensor<*xi1>, tensor<*xi32>, tensor<*xf32>)
  func.return
}

// -----

func.func @parallel_execute_regions_with_data_results(%arg0: tensor<i32>) -> tensor<i32> {
  // expected-error @+1 {{'func' attribute refers to an undefined function: undefined_func}}
  %0 = "tf_device.cluster_func"(%arg0) {func = @undefined_func} : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
