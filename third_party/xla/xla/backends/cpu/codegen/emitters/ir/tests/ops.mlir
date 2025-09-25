// RUN: emitters_opt %s --split-input-file -verify-roundtrip

func.func @load(%arg0: !xla_cpu.call_frame) -> tensor<32x32xf32> {
  %0 = xla_cpu.load %arg0, 0 : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

func.func @load(%arg0: !xla_cpu.call_frame) -> memref<64x32xf32> {
  %0 = xla_cpu.load %arg0, 0 : memref<64x32xf32>
  return %0 : memref<64x32xf32>
}

// -----

func.func @extract_workgroup_id(%arg0: !xla_cpu.call_frame) -> index {
  %0 = xla_cpu.extract_workgroup_id %arg0, x
  return %0 : index
}
