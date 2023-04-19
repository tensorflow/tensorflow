// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_disassemble(
    %arg0: !ifrt.array<tensor<2x4xi32>, "shard0", [0,1]>) {
  %0, %1 = "ifrt.Disassemble"(%arg0)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x4xi32>, "shard0", [0,1]>)
      -> (!ifrt.array<tensor<2x2xi32>, "shard1", [0]>,
          !ifrt.array<tensor<2x2xi32>, "shard2", [1]>)
  return
}

// -----

func.func @disassemble_requires_outputs_on_single_devices(
    %arg0: !ifrt.array<tensor<2x4xi32>, "shard0", [0,1,2]>) {
  // expected-error@+1 {{'ifrt.Disassemble' op requires every output to be a single device array. Actual: '!ifrt.array<tensor<2x2xi32>, "shard1", [0, 1]>'}}
  %0, %1 = "ifrt.Disassemble"(%arg0)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x4xi32>, "shard0", [0,1,2]>)
      -> (!ifrt.array<tensor<2x2xi32>, "shard1", [0,1]>,
          !ifrt.array<tensor<2x2xi32>, "shard2", [2]>)
  return
}

// -----

func.func @disassemble_requires_same_device_list(
    %arg0: !ifrt.array<tensor<2x4xi32>, "shard0", [0,1]>) {
  // expected-error@+1 {{'ifrt.Disassemble' op requires the same input/output device list. Input 0, 1 vs Output 1, 2}}
  %0, %1 = "ifrt.Disassemble"(%arg0)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x4xi32>, "shard0", [0,1]>)
      -> (!ifrt.array<tensor<2x2xi32>, "shard1", [1]>,
          !ifrt.array<tensor<2x2xi32>, "shard2", [2]>)
  return
}
