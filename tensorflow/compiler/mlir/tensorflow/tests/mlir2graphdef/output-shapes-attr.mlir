// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<10xi32>) -> tensor<10xi32>
attributes {tf.entry_function = {inputs = "input0", outputs = "output0"}} {
  %graph = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<10xi32>
  }
  return %graph : tensor<10xi32>
}

// CHECK:      node {
// CHECK-NEXT:   name: "input0"
// CHECK-NEXT:   op: "_Arg"
// CHECK:          key: "T"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK:          key: "_output_shapes"
// CHECK-NEXT:     value {
// CHECK-NEXT:       list {
// CHECK-NEXT:         shape {
// CHECK-NEXT:           dim {
// CHECK-NEXT:             size: 10
// CHECK:        name: "output0"
// CHECK-NEXT:   op: "_Retval"
// CHECK-NEXT:   input: "input0"
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "T"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "index"
// CHECK-NEXT:     value {
// CHECK-NEXT:       i: 0
// CHECK-NEXT:     }
// CHECK-NEXT:   }

func private @simple_callee(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %graph = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<2xi32>
  }
  return %graph#0 : tensor<2xi32>
}

// Test with a TPUPartitionedCallOp that is not inlined.
func @dont_inline_tpu_partitioned_call(%arg0: tensor<2xi32> {tf._user_specified_name = "inputs_0"}, %arg1: tensor<2xi32> {tf._user_specified_name = "inputs_1"}) -> tensor<2xi32> {
  %graph = tf_executor.graph {
    %result:2 = tf_executor.island wraps "tf.TPUPartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @simple_callee} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    tf_executor.fetch %result#0 : tensor<2xi32>
  }
  return %graph : tensor<2xi32>
}

// CHECK:       library {
// CHECK-NEXT:    function {
// CHECK-NEXT:      signature {
// CHECK-NEXT:        name: "simple_callee"
// CHECK:         function {
// CHECK-NEXT:      signature {
// CHECK-NEXT:        name: "dont_inline_tpu_partitioned_call"
// CHECK-NEXT:        input_arg {
// CHECK-NEXT:          name: "dont_inline_tpu_partitioned_call"
// CHECK-NEXT:          type: DT_INT32
// CHECK-NEXT:        }
// CHECK-NEXT:        input_arg {
// CHECK-NEXT:          name: "dont_inline_tpu_partitioned_call1"
// CHECK-NEXT:          type: DT_INT32
// CHECK-NEXT:        }
// CHECK:           arg_attr {
// CHECK-NEXT:        key: 0
// CHECK-NEXT:        value {
// CHECK-NEXT:          attr {
// CHECK-NEXT:            key: "_output_shapes"
// CHECK-NEXT:            value {
// CHECK-NEXT:              list {
// CHECK-NEXT:                shape {
// CHECK-NEXT:                  dim {
// CHECK-NEXT:                    size: 2
// CHECK:               attr {
// CHECK-NEXT:            key: "_user_specified_name"
// CHECK-NEXT:            value {
// CHECK-NEXT:              s: "inputs_0"
// CHECK:           arg_attr {
// CHECK-NEXT:        key: 1
// CHECK-NEXT:        value {
// CHECK-NEXT:          attr {
// CHECK-NEXT:            key: "_output_shapes"
// CHECK-NEXT:            value {
// CHECK-NEXT:              list {
// CHECK-NEXT:                shape {
// CHECK-NEXT:                  dim {
// CHECK-NEXT:                    size: 2
// CHECK:               attr {
// CHECK-NEXT:            key: "_user_specified_name"
// CHECK-NEXT:            value {
// CHECK-NEXT:              s: "inputs_1"
