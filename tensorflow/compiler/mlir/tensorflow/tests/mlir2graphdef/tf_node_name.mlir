// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

// CHECK: node {
// CHECK-NEXT:  name: "Const"
// CHECK-NEXT:  op: "Const"
// CHECK-NEXT:  attr {
// CHECK-NEXT:    key: "dtype"
// CHECK-NEXT:    value {
// CHECK-NEXT:      type: DT_INT32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  attr {
// CHECK-NEXT:    key: "value"
// CHECK-NEXT:    value {
// CHECK-NEXT:      tensor {
// CHECK-NEXT:        dtype: DT_INT32
// CHECK-NEXT:        tensor_shape {
// CHECK-NEXT:        }
// CHECK-NEXT:        int_val: 1
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }

func @main() {
    %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Const", value = dense<1> : tensor<i32>} : () -> (tensor<i32>, !_tf.control)
    return
}
