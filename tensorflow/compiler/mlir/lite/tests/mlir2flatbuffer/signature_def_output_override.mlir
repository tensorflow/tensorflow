// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

// CHECK: {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ {
// CHECK-NEXT:    deprecated_builtin_code: 9,
// CHECK-NEXT:    version: 1,
// CHECK-NEXT:    builtin_code: FULLY_CONNECTED
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 1, 3 ],
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "serving_default_input2:0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      shape_signature: [ -1, 3 ],
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 3 ],
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "serving_default_input1:0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      shape_signature: [ -1, 3 ],
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 5 ],
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "arith.constant",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 5, 3 ],
// CHECK-NEXT:      buffer: 4,
// CHECK-NEXT:      name: "arith.constant1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 5 ],
// CHECK-NEXT:      buffer: 5,
// CHECK-NEXT:      name: "StatefulPartitionedCall:1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      shape_signature: [ -1, 5 ],
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1 ],
// CHECK-NEXT:    outputs: [ 4, 4 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 3, 2 ],
// CHECK-NEXT:      outputs: [ 4 ],
// CHECK-NEXT:      builtin_options_type: FullyConnectedOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63 ]
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 49, 46, 53, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:    name: "min_runtime_version",
// CHECK-NEXT:    buffer: 6
// CHECK-NEXT:  } ],
// CHECK-NEXT:  signature_defs: [ {
// CHECK-NEXT:    inputs: [ {
// CHECK-NEXT:      name: "input1",
// CHECK-NEXT:      tensor_index: 1
// CHECK-NEXT:    }, {
// CHECK-NEXT:      name: "input2"
// CHECK-NEXT:    } ],
// CHECK-NEXT:    outputs: [ {
// CHECK-NEXT:      name: "end_logits",
// CHECK-NEXT:      tensor_index: 4
// CHECK-NEXT:    }, {
// CHECK-NEXT:      name: "start_logits",
// CHECK-NEXT:      tensor_index: 4
// CHECK-NEXT:    } ],
// CHECK-NEXT:    signature_key: "serving_default"
// CHECK-NEXT:  } ]
// CHECK-NEXT:}
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 554 : i32}, tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<?x3xf32> {tf_saved_model.index_path = ["input2"]}, %arg1: tensor<?x3xf32> {tf_saved_model.index_path = ["input1"]}) -> (tensor<?x5xf32> {tf_saved_model.index_path = ["start_logits"]}, tensor<?x5xf32> {tf_saved_model.index_path = ["end_logits"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input2:0,serving_default_input1:0", outputs = "StatefulPartitionedCall:1,StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<0.000000e+00> : tensor<5xf32>
    %cst_0 = arith.constant dense<1.0> : tensor<5x3xf32>
    %0 = "tfl.fully_connected"(%arg0, %cst_0, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<?x3xf32>, tensor<5x3xf32>, tensor<5xf32>) -> tensor<?x5xf32>
    func.return %0, %0 : tensor<?x5xf32>, tensor<?x5xf32>
  }
}
