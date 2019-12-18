// RUN: tf-mlir-translate -mlir-to-graphdef %s -tf-graph-as-function -o - | FileCheck %s --dump-input=fail

func @main(%arg0: tensor<*x!tf.resource>, %arg1: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>, %arg2: tensor<*xf32>, %arg3: tensor<2x4x6x8xi32>) -> (tensor<f32>, tensor<f32>)
attributes {tf.entry_function = {inputs = "args_0,args_1,args_2,args_3", outputs = "rets_0_RetVal,rets_1_RetVal"}} {
  %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", name = "const", value = dense<0.000000e+00> : tensor<f32>} : () -> (tensor<f32>, !_tf.control)
  %1:2 = "_tf.Identity"(%0#0) {T = "tfdtype$DT_FLOAT", device = "", name = "identity"} : (tensor<f32>) -> (tensor<f32>, !_tf.control)
  %2:2 = "_tf.StatefulPartitionedCall"(%0#0, %arg1) {Tin = ["tfdtype$DT_FLOAT", "tfdtype$DT_RESOURCE"], Tout = ["tfdtype$DT_FLOAT"], _gradient_op_type = "PartitionedCall-1205", config = "", config_proto = "\0A\07\0A\03GPU\10\00\0A\07\0A\03CPU\10\012\02J\008\01", device = "", executor_type = "", f = @function0, name = "statefulpartitionedcall"} : (tensor<f32>, tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>) -> (tensor<f32>, !_tf.control)
  return %1#0, %2#0 : tensor<f32>, tensor<f32>
}

func @function0(%arg0: tensor<*xf32>, %arg1: tensor<*x!tf.resource>) -> tensor<*xf32>
attributes {tf.signature.is_stateful} {
  %0:2 = "_tf.Identity"(%arg0) {T = "tfdtype$DT_FLOAT", device = "", name = "Identity"} : (tensor<*xf32>) -> (tensor<*xf32>, !_tf.control)
  return %0#0 : tensor<*xf32>
}

// CHECK:      node {
// CHECK-NEXT:   name: "args_0"
// CHECK-NEXT:   op: "_Arg"
// CHECK:          i: 0
// CHECK:      node {
// CHECK-NEXT:   name: "args_1"
// CHECK-NEXT:   op: "_Arg"
// CHECK:          i: 1
// CHECK:      node {
// CHECK-NEXT:   name: "args_2"
// CHECK-NEXT:   op: "_Arg"
// CHECK:          i: 2
// CHECK:      node {
// CHECK-NEXT:   name: "args_3"
// CHECK-NEXT:   op: "_Arg"
// CHECK:          i: 3
// CHECK:      node {
// CHECK-NEXT:   name: "const"
// CHECK-NEXT:   op: "Const"
// CHECK:      node {
// CHECK-NEXT:   name: "identity"
// CHECK-NEXT:   op: "Identity"
// CHECK-NEXT:   input: "const"
// CHECK:      node {
// CHECK-NEXT:   name: "statefulpartitionedcall"
// CHECK-NEXT:   op: "StatefulPartitionedCall"
// CHECK-NEXT:   input: "const"
// CHECK-NEXT:   input: "args_1"
// CHECK:          key: "f"
// CHECK-NEXT:     value {
// CHECK-NEXT:       func {
// CHECK-NEXT:         name: "function0"
// CHECK:      node {
// CHECK-NEXT:   name: "rets_0_RetVal"
// CHECK-NEXT:   op: "_Retval"
// CHECK-NEXT:   input: "identity"
// CHECK:      node {
// CHECK-NEXT:   name: "rets_1_RetVal"
// CHECK-NEXT:   op: "_Retval"
// CHECK-NEXT:   input: "statefulpartitionedcall"
// CHECK:      library {
// CHECK-NEXT:   function {
// CHECK-NEXT:     signature {
// CHECK-NEXT:       name: "function0"
// CHECK-NEXT:       input_arg {
// CHECK-NEXT:         name: "function0"
// CHECK:            input_arg {
// CHECK-NEXT:         name: "function01"
// CHECK:            output_arg {
// CHECK-NEXT:         name: "function02"
// CHECK:          node_def {
// CHECK-NEXT:       name: "Identity"
// CHECK-NEXT:       op: "Identity"
// CHECK-NEXT:       input: "function0"
// CHECK:          ret {
// CHECK-NEXT:       key: "function02"
// CHECK-NEXT:       value: "Identity:output:0"
