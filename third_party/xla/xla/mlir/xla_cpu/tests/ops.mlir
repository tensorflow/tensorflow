// RUN: xla-cpu-opt %s -split-input-file -empty-tensor-to-alloc-tensor \
// RUN:   -one-shot-bufferize | FileCheck %s

func.func @memref_cast(%arg0: memref<10xf32>) -> memref<10xi32> {
  %ret = xla_cpu.memref_element_cast %arg0 : memref<10xf32> to memref<10xi32>
  return %ret : memref<10xi32>
}

// CHECK: xla_cpu.memref_element_cast {{.*}} : memref<10xf32> to memref<10xi32>

func.func @memref_cast_i1(%arg0: memref<10xi1>) -> memref<10xi8> {
  %ret = xla_cpu.memref_element_cast %arg0 : memref<10xi1> to memref<10xi8>
  return %ret : memref<10xi8>
}

// CHECK: xla_cpu.memref_element_cast {{.*}} : memref<10xi1> to memref<10xi8>