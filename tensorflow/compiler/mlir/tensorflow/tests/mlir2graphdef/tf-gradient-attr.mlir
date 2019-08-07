// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
// CHECK:      node {
// CHECK-NEXT:   name: "Const"
// CHECK-NEXT:   op: "Const"
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "dtype"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_FLOAT
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "value"
// CHECK-NEXT:     value {
// CHECK-NEXT:       tensor {
// CHECK-NEXT:         dtype: DT_FLOAT
// CHECK-NEXT:         tensor_shape {
// CHECK-NEXT:         }
// CHECK-NEXT:         float_val: 0.25
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   experimental_debug_info {
// CHECK-NEXT:   }
// CHECK-NEXT: }
  %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", value = dense<2.500000e-01> : tensor<f32>} : () -> (tensor<f32>, !_tf.control) loc("Const")

// CHECK:      node {
// CHECK-NEXT:   name: "foo"
// CHECK-NEXT:   op: "foo"
// CHECK-NEXT:   input: "Const"
// CHECK-NEXT:   experimental_debug_info {
// CHECK-NEXT:   }
// CHECK-NEXT: }
  %1:2 = "_tf.foo"(%0#0) {device = ""} : (tensor<f32>) -> (tensor<*xf32>, !_tf.control) loc("foo")
  return
}

// CHECK:      library {
// CHECK-NEXT:   function {
// CHECK-NEXT:     signature {
// CHECK-NEXT:       name: "foo"
// CHECK-NEXT:       input_arg {
// CHECK-NEXT:         name: "foo"
// CHECK-NEXT:         type: DT_FLOAT
// CHECK-NEXT:       }
// CHECK-NEXT:       output_arg {
// CHECK-NEXT:         name: "foo1"
// CHECK-NEXT:         type: DT_FLOAT
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     ret {
// CHECK-NEXT:       key: "foo1"
// CHECK-NEXT:       value: "foo"
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   function {
// CHECK-NEXT:     signature {
// CHECK-NEXT:       name: "foo_grad"
// CHECK-NEXT:       input_arg {
// CHECK-NEXT:         name: "foo_grad"
// CHECK-NEXT:         type: DT_FLOAT
// CHECK-NEXT:       }
// CHECK-NEXT:       input_arg {
// CHECK-NEXT:         name: "foo_grad1"
// CHECK-NEXT:         type: DT_FLOAT
// CHECK-NEXT:       }
// CHECK-NEXT:       output_arg {
// CHECK-NEXT:         name: "foo_grad2"
// CHECK-NEXT:         type: DT_FLOAT
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     ret {
// CHECK-NEXT:       key: "foo_grad2"
// CHECK-NEXT:       value: "foo_grad"
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   gradient {
// CHECK-NEXT:     function_name: "foo"
// CHECK-NEXT:     gradient_func: "foo_grad"
// CHECK-NEXT:   }
// CHECK-NEXT: }
func @foo_grad(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  return %arg0 : tensor<*xf32>
}

func @foo(%arg0: tensor<*xf32>) -> tensor<*xf32>
  attributes  {tf.gradient = @foo_grad} {
  return %arg0 : tensor<*xf32>
}

