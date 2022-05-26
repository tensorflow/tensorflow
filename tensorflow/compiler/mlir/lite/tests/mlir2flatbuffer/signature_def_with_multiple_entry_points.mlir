// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:   operator_codes: [ {
// CHECK-NEXT:     version: 1
// CHECK-NEXT:   }, {
// CHECK-NEXT:     deprecated_builtin_code: 41,
// CHECK-NEXT:     version: 1,
// CHECK-NEXT:     builtin_code: SUB
// CHECK-NEXT:   } ],
// CHECK-NEXT:   subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 1 ],
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "input1:0",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      shape_signature: [ -1 ],
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1 ],
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "input2:0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      shape_signature: [ -1 ],
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1 ],
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "tfl.add",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      shape_signature: [ -1 ],
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1 ],
// CHECK-NEXT:      buffer: 4,
// CHECK-NEXT:      name: "result:0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      shape_signature: [ -1 ],
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1 ],
// CHECK-NEXT:    outputs: [ 3 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 1 ],
// CHECK-NEXT:      outputs: [ 2 ],
// CHECK-NEXT:      builtin_options_type: AddOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      inputs: [ 2, 2 ],
// CHECK-NEXT:      outputs: [ 3 ],
// CHECK-NEXT:      builtin_options_type: AddOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    name: "add"
// CHECK-NEXT:  }, {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 1 ],
// CHECK-NEXT:      buffer: 5,
// CHECK-NEXT:      name: "input2:0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      shape_signature: [ -1 ],
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1 ],
// CHECK-NEXT:      buffer: 6,
// CHECK-NEXT:      name: "input1:0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      shape_signature: [ -1 ],
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1 ],
// CHECK-NEXT:      buffer: 7,
// CHECK-NEXT:      name: "result:0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      shape_signature: [ -1 ],
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1 ],
// CHECK-NEXT:    outputs: [ 2 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      opcode_index: 1,
// CHECK-NEXT:      inputs: [ 0, 1 ],
// CHECK-NEXT:      outputs: [ 2 ],
// CHECK-NEXT:      builtin_options_type: SubOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    name: "sub"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 49, 46, 54, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:    name: "min_runtime_version",
// CHECK-NEXT:    buffer: 8
// CHECK-NEXT:  } ],
// CHECK-NEXT:  signature_defs: [ {
// CHECK-NEXT:    inputs: [ {
// CHECK-NEXT:      name: "input1"
// CHECK-NEXT:    }, {
// CHECK-NEXT:      name: "input2",
// CHECK-NEXT:      tensor_index: 1
// CHECK-NEXT:    } ],
// CHECK-NEXT:    outputs: [ {
// CHECK-NEXT:      name: "result",
// CHECK-NEXT:      tensor_index: 3
// CHECK-NEXT:    } ],
// CHECK-NEXT:    signature_key: "add"
// CHECK-NEXT:  }, {
// CHECK-NEXT:    inputs: [ {
// CHECK-NEXT:      name: "input1",
// CHECK-NEXT:      tensor_index: 1
// CHECK-NEXT:    }, {
// CHECK-NEXT:      name: "input2"
// CHECK-NEXT:    } ],
// CHECK-NEXT:    outputs: [ {
// CHECK-NEXT:      name: "result",
// CHECK-NEXT:      tensor_index: 2
// CHECK-NEXT:    } ],
// CHECK-NEXT:    signature_key: "sub",
// CHECK-NEXT:    subgraph_index: 1
// CHECK-NEXT:  } ]
// CHECK-NEXT: }
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 554 : i32}, tf_saved_model.semantics} {
  func.func @add(%arg0: tensor<?xf32> {tf_saved_model.index_path = ["input1"]}, %arg1: tensor<?xf32> {tf_saved_model.index_path = ["input2"]}) -> (tensor<?xf32> {tf_saved_model.index_path = ["result"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "input1:0,input2:0", outputs = "result:0"}, tf_saved_model.exported_names = ["add"]} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<?xf32>
    %1 = tfl.add %0, %0 {fused_activation_function = "NONE"} : tensor<?xf32>
    func.return %1 : tensor<?xf32>
  }

  func.func @sub(%arg0: tensor<?xf32> {tf_saved_model.index_path = ["input2"]}, %arg1: tensor<?xf32> {tf_saved_model.index_path = ["input1"]}) -> (tensor<?xf32> {tf_saved_model.index_path = ["result"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "input2:0,input1:0", outputs = "result:0"}, tf_saved_model.exported_names = ["sub"]} {
    %0 = tfl.sub %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<?xf32>
    func.return %0 : tensor<?xf32>
  }
}
