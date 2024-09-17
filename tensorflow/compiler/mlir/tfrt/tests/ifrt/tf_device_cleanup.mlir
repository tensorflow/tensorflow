// RUN: tf-tfrt-opt %s -tf-device-cleanup | FileCheck %s

// CHECK-LABEL: func @ops_with_device
func.func @ops_with_device() {
  %0 = "tf.VarHandleOp"() {container = "", shared_name = "var", device = "/device/..."} : () -> tensor<!tf_type.resource<tensor<1xf32>>>
  // CHECK-NOT: device = "/device/..."
  func.return
}
