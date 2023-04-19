// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_assemble(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0]>,
    %arg1: !ifrt.array<tensor<2x2xi32>, "shard1", [1]>) {
  %0 = "ifrt.Assemble"(%arg0, %arg1)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x2xi32>, "shard0", [0]>,
         !ifrt.array<tensor<2x2xi32>, "shard1", [1]>)
      -> !ifrt.array<tensor<2x4xi32>, "shard2", [0,1]>
  return
}

// -----

func.func @assemble_requires_inputs_on_single_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>,
    %arg1: !ifrt.array<tensor<2x2xi32>, "shard1", [2]>) {
  // expected-error@+1 {{'ifrt.Assemble' op requires every input to be a single device array. Actual: '!ifrt.array<tensor<2x2xi32>, "shard0", [0, 1]>'}}
  %0 = "ifrt.Assemble"(%arg0, %arg1)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>,
         !ifrt.array<tensor<2x2xi32>, "shard1", [2]>)
      -> !ifrt.array<tensor<2x4xi32>, "shard2", [0,1,2]>
  return
}

// -----

func.func @assemble_requires_same_device_list(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0]>,
    %arg1: !ifrt.array<tensor<2x2xi32>, "shard1", [1]>) {
  // expected-error@+1 {{'ifrt.Assemble' op requires the same input/output device list. Input 0, 1 vs Output 1, 2}}
  %0 = "ifrt.Assemble"(%arg0, %arg1)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x2xi32>, "shard0", [0]>,
         !ifrt.array<tensor<2x2xi32>, "shard1", [1]>)
      -> !ifrt.array<tensor<2x4xi32>, "shard2", [1,2]>
  return
}
