// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

// CHECK: {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ {
// CHECK-NEXT:    deprecated_builtin_code: 127,
// CHECK-NEXT:    version: 1,
// CHECK-NEXT:    builtin_code: STABLEHLO_COMPOSITE
// CHECK-NEXT:  }, {
// CHECK-NEXT:    version: 1
// CHECK-NEXT:  }, {
// CHECK-NEXT:    deprecated_builtin_code: 41,
// CHECK-NEXT:    version: 1,
// CHECK-NEXT:    builtin_code: SUB
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "arg0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "arg1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [  ],
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "arith.constant",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [  ],
// CHECK-NEXT:      buffer: 4,
// CHECK-NEXT:      name: "arith.constant1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 5,
// CHECK-NEXT:      name: "vhlo.composite_v1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 6,
// CHECK-NEXT:      name: "vhlo.composite_v11",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 7,
// CHECK-NEXT:      name: "tfl.add",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 8,
// CHECK-NEXT:      name: "tfl.sub",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1 ],
// CHECK-NEXT:    outputs: [ 7 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 1 ],
// CHECK-NEXT:      outputs: [ 4 ],
// CHECK-NEXT:      builtin_options_2_type: StableHLOCompositeOptions,
// CHECK-NEXT:      builtin_options_2: {
// CHECK-NEXT:        name: "test.TEST_COMPOSITE",
// CHECK-NEXT:        decomposition_subgraph_index: 2,
// CHECK-NEXT:        composite_attributes: [ 0, 0, 1, 0, 0, 36, 1 ]
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      inputs: [ 4, 1 ],
// CHECK-NEXT:      outputs: [ 5 ],
// CHECK-NEXT:      builtin_options_2_type: StableHLOCompositeOptions,
// CHECK-NEXT:      builtin_options_2: {
// CHECK-NEXT:        name: "test.TEST_COMPOSITE",
// CHECK-NEXT:        decomposition_subgraph_index: 1,
// CHECK-NEXT:        composite_attributes: [ 0, 0, 1, 0, 0, 36, 1 ]
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 1,
// CHECK-NEXT:      inputs: [ 5, 2 ],
// CHECK-NEXT:      outputs: [ 6 ],
// CHECK-NEXT:      builtin_options_type: AddOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 2,
// CHECK-NEXT:      inputs: [ 6, 3 ],
// CHECK-NEXT:      outputs: [ 7 ],
// CHECK-NEXT:      builtin_options_type: SubOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:  }, {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 9,
// CHECK-NEXT:      name: "XlaCallModule_test.TEST_COMPOSITE.impl_0_arg0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 10,
// CHECK-NEXT:      name: "XlaCallModule_test.TEST_COMPOSITE.impl_0_arg1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [  ],
// CHECK-NEXT:      buffer: 11,
// CHECK-NEXT:      name: "arith.constant2",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 12,
// CHECK-NEXT:      name: "tfl.add1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 13,
// CHECK-NEXT:      name: "tfl.add2",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1 ],
// CHECK-NEXT:    outputs: [ 4 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      opcode_index: 1,
// CHECK-NEXT:      inputs: [ 0, 1 ],
// CHECK-NEXT:      outputs: [ 3 ],
// CHECK-NEXT:      builtin_options_type: AddOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 1,
// CHECK-NEXT:      inputs: [ 3, 2 ],
// CHECK-NEXT:      outputs: [ 4 ],
// CHECK-NEXT:      builtin_options_type: AddOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    name: "XlaCallModule_test.TEST_COMPOSITE.impl_0"
// CHECK-NEXT:  }, {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 14,
// CHECK-NEXT:      name: "XlaCallModule_test.TEST_COMPOSITE.impl_0_0_arg0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 15,
// CHECK-NEXT:      name: "XlaCallModule_test.TEST_COMPOSITE.impl_0_0_arg1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [  ],
// CHECK-NEXT:      buffer: 11,
// CHECK-NEXT:      name: "arith.constant3",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 17,
// CHECK-NEXT:      name: "tfl.add3",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 10 ],
// CHECK-NEXT:      buffer: 18,
// CHECK-NEXT:      name: "tfl.add4",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1 ],
// CHECK-NEXT:    outputs: [ 4 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      opcode_index: 1,
// CHECK-NEXT:      inputs: [ 0, 1 ],
// CHECK-NEXT:      outputs: [ 3 ],
// CHECK-NEXT:      builtin_options_type: AddOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 1,
// CHECK-NEXT:      inputs: [ 3, 2 ],
// CHECK-NEXT:      outputs: [ 4 ],
// CHECK-NEXT:      builtin_options_type: AddOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    name: "XlaCallModule_test.TEST_COMPOSITE.impl_0_0"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 32, 65 ]
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 160, 65 ]
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
// CHECK-NEXT:    data: [ 0, 0, 200, 66 ]
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
// CHECK-NEXT:    data: [ 50, 46, 49, 55, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:    name: "min_runtime_version",
// CHECK-NEXT:    buffer: 19
// CHECK-NEXT:  } ],
// CHECK-NEXT:  signature_defs: [  ]
// CHECK-NEXT:}

func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> (tensor<10xf32>) {
  %cst = arith.constant dense<1.000000e+01> : tensor<f32>
  %cst_0 = arith.constant dense<2.000000e+01> : tensor<f32>
  %0 = "vhlo.composite_v1"(%arg0, %arg1) <{composite_attributes = #vhlo.dict_v1<{}>, decomposition = #vhlo.string_v1<"XlaCallModule_test.TEST_COMPOSITE.impl_0_0">, name = #vhlo.string_v1<"test.TEST_COMPOSITE">, version = #vhlo.integer_v1<0 : i64>}> : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %1 = "vhlo.composite_v1"(%0, %arg1) <{composite_attributes = #vhlo.dict_v1<{}>, decomposition = #vhlo.string_v1<"XlaCallModule_test.TEST_COMPOSITE.impl_0">, name = #vhlo.string_v1<"test.TEST_COMPOSITE">, version = #vhlo.integer_v1<0 : i64>}> : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %2 = tfl.add(%1, %cst) <{fused_activation_function = "NONE"}> : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
  %3 = tfl.sub(%2, %cst_0) <{fused_activation_function = "NONE"}> : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
  return %3 : tensor<10xf32>
}
func.func private @XlaCallModule_test.TEST_COMPOSITE.impl_0(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %cst = arith.constant dense<1.000000e+02> : tensor<f32>
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<10xf32>
  %1 = tfl.add(%0, %cst) <{fused_activation_function = "NONE"}> : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}
func.func private @XlaCallModule_test.TEST_COMPOSITE.impl_0_0(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %cst = arith.constant dense<1.000000e+02> : tensor<f32>
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<10xf32>
  %1 = tfl.add(%0, %cst) <{fused_activation_function = "NONE"}> : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}
