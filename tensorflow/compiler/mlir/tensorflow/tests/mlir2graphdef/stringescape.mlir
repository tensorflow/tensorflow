// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
^bb0:
  // CHECK:      key: "listvalue"
  // CHECK-NEXT: value {
  // CHECK-NEXT:   list {
  // CHECK-NEXT:     s: " \n"
  // CHECK-NEXT:   }
  // CHECK:      key: "value"
  // CHECK-NEXT: value {
  // CHECK-NEXT:   s: " 0\n\000\000"
  %0:2 = "_tf.Empty"() {name = "dummy", dtype = "tfdtype$DT_INT32", value = "\200\n\00\00", listvalue = ["\20\0A"]} : () -> (tensor<2xi32>, !_tf.control)
  return
}

