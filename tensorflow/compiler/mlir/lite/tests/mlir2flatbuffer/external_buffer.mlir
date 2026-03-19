// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

module {
  func.func public @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "tfl.external_const"() <{external_buffer = #tfl.external_buffer<group_name = "test.bin", offset = 0, length = 13, packing = "unpacked">}> : () -> tensor<2x2xf32>
    %1 = tfl.add %arg0, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}

// CHECK:  tensors: [ {
// CHECK:    shape: [ 2, 2 ],
// CHECK:    buffer: 1,
// CHECK:    name: "arg0",
// CHECK:    has_rank: true
// CHECK:  }, {
// CHECK:    shape: [ 2, 2 ],
// CHECK:    name: "tfl.external_const",
// CHECK:    has_rank: true,
// CHECK:    external_buffer: 2147483648
// CHECK:  }, {
// CHECK:    shape: [ 2, 2 ],
// CHECK:    buffer: 2,
// CHECK:    name: "tfl.add",
// CHECK:    has_rank: true
// CHECK:  } ],
// CHECK:  external_buffer_groups: [ {
// CHECK:    name: "test.bin"
// CHECK:  } ],
// CHECK:  external_buffers: [ {
// CHECK:    id: 2147483648,
// CHECK:    length: 13,
// CHECK:    packing: "unpacked"
// CHECK:  } ]
