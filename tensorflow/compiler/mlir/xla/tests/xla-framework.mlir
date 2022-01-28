// RUN: xla-opt %s | xla-opt | FileCheck %s

// CHECK-LABEL: @buffer_type
func @buffer_type(%arg1: !xla_framework.buffer) -> !xla_framework.buffer
                      attributes {xla_entry} {
  return %arg1 : !xla_framework.buffer
}

// CHECK-LABEL: @mem_to_buffer
func @mem_to_buffer(%arg : memref<f32>) -> !xla_framework.buffer
                      attributes {xla_entry} {
  %result = xla_framework.mem_to_buffer %arg : memref<f32>
  return %result : !xla_framework.buffer
}

// CHECK-LABEL: @buffer
func @buffer(%arg : !xla_framework.buffer) -> memref<f32>
                      attributes {xla_entry} {
  %result = xla_framework.buffer_to_mem %arg : memref<f32>
  return %result : memref<f32>
}
