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
