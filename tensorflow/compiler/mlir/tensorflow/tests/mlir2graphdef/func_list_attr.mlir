// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
  tf_executor.graph {
// CHECK:      node {
// CHECK-NEXT:   name: "predicate"
// CHECK-NEXT:   op: "Const"
// CHECK-NEXT:   attr {
// CHECK:          key: "dtype"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "value"
// CHECK-NEXT:     value {
// CHECK-NEXT:       tensor {
// CHECK-NEXT:         dtype: DT_INT32
// CHECK-NEXT:         tensor_shape {
// CHECK-NEXT:         }
// CHECK-NEXT:         int_val: 0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK:      }
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> tensor<i32> loc("predicate")

// CHECK:      node {
// CHECK-NEXT:   name: "Case"
// CHECK-NEXT:   op: "Case"
// CHECK-NEXT:   input: "predicate"
// CHECK:        attr {
// CHECK:          key: "branches"
// CHECK-NEXT:     value {
// CHECK-NEXT:       list {
// CHECK-NEXT:         func {
// CHECK-NEXT:           name: "foo"
// CHECK-NEXT:         }
// CHECK-NEXT:         func {
// CHECK-NEXT:           name: "bar"
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK:      }
    %1:2 = tf_executor.island wraps "tf.Case"(%0#0) {Tin = [], Tout = ["tfdtype$DT_FLOAT"], branches = [@foo, @bar], device = "", output_shapes = [], is_stateless = false} : (tensor<i32>) -> tensor<*xf32> loc("Case")
    tf_executor.fetch
  }
  return
}

// CHECK-DAG: name: "foo"
func @foo() -> tensor<10xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", value = dense<1.000000e+00> : tensor<10xf32>} : () -> tensor<10xf32> loc("const_1")
    tf_executor.fetch %1#0 : tensor<10xf32>
  }
  return %0 : tensor<10xf32>
}

// CHECK-DAG: name: "bar"
func @bar() -> tensor<10xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", value = dense<2.000000e+00> : tensor<10xf32>} : () -> tensor<10xf32> loc("const_2")
    tf_executor.fetch %1#0 : tensor<10xf32>
  }
  return %0 : tensor<10xf32>
}
