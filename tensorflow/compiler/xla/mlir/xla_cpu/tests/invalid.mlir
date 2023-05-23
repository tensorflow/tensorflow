// RUN: xla-cpu-opt %s -split-input-file -verify-diagnostics

func.func @memref_cast_out_of_place(%arg0: memref<10xi1>) -> memref<10xi16> {
  // expected-error @+1 {{cannot cast from 'i1' to 'i16'}}
  %ret = xla_cpu.memref_element_cast %arg0 : memref<10xi1> to memref<10xi16>
  return %ret : memref<10xi16>
}
