// RUN: tf-mlir-translate -analyze-tf-for-tfrt %s | FileCheck %s

func @main(%serialized: tensor<32x!tf.string>,
           %names : tensor<32x!tf.string>,
           %dense_keys : tensor<2x!tf.string>,
           %dense_default_0 : tensor<?xf32>,
           %dense_default_1 : tensor<?xf32>) {
  // CHECK:      summary {
  // CHECK-NEXT:   ref_variable: true
  // CHECK-NEXT:   incompatible_variable: true
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT:   key: "tf.AssignVariableOp"
  // CHECK-NEXT:   value {
  // CHECK-NEXT:     count: 1
  // CHECK-NEXT:     report {
  // CHECK-NEXT:       incompatible_variable: true
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT:   key: "tf.Const"
  // CHECK-NEXT:   value {
  // CHECK-NEXT:     count: 2
  // CHECK-NEXT:     report {
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT:   key: "tf.ParseExampleV2"
  // CHECK-NEXT:   value {
  // CHECK-NEXT:     count: 1
  // CHECK-NEXT:     report {
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT:   key: "tf.VarHandleOp"
  // CHECK-NEXT:   value {
  // CHECK-NEXT:     count: 1
  // CHECK-NEXT:     report {
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT:   key: "tf.VariableV2"
  // CHECK-NEXT:   value {
  // CHECK-NEXT:     count: 1
  // CHECK-NEXT:     report {
  // CHECK-NEXT:       ref_variable: true
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  %0 = "tf.VariableV2"() {shape = #tf.shape<2>, container = "", shared_name = ""} : () -> tensor<!tf.int32ref>
  %1 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %2 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>
  "tf.AssignVariableOp"(%2, %1) : (tensor<*x!tf.resource>, tensor<f32>) -> ()
  %empty_str_vector = "tf.Const"()
    {dtype = !tf.string, value = opaque<"tf", "0x746674656E736F722464747970653A2044545F535452494E472074656E736F725F7368617065207B2064696D207B207D207D"> : tensor<0x!tf.string>}
      : () -> tensor<0x!tf.string>
  %result:2 = "tf.ParseExampleV2"(%serialized, %names, %empty_str_vector, %dense_keys, %empty_str_vector, %dense_default_0, %dense_default_1)
    {dense_shapes = [#tf.shape<>, #tf.shape<>], num_sparse = 0 : i64, result_segment_sizes = dense<[0, 0, 0, 2, 0, 0]> : vector<6xi32>}
      : (tensor<32x!tf.string>, tensor<32x!tf.string>, tensor<0x!tf.string>, tensor<2x!tf.string>, tensor<0x!tf.string>, tensor<?xf32>, tensor<?xf32>) -> (tensor<32xf32>, tensor<32xf32>)
  return
}
