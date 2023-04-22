// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
  tf_executor.graph {
    // CHECK:       key: "emptylist"
    // CHECK-NEXT:  value {
    // CHECK-NEXT:    list {
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    // CHECK:       key: "typelist"
    // CHECK-NEXT:  value {
    // CHECK-NEXT:    list {
    // CHECK-NEXT:      type: DT_INT32
    // CHECK-NEXT:      type: DT_FLOAT
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    %0:2 = tf_executor.island wraps "tf.Placeholder"() {name = "dummy", dtype = "tfdtype$DT_FLOAT", emptylist = [], typelist = ["tfdtype$DT_INT32", "tfdtype$DT_FLOAT"]} : () -> tensor<*xi32>
    tf_executor.fetch
  }
  return
}
