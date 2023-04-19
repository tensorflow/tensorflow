// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_disassemble(
    %arg0: !ifrt.array<tensor<2x4xi32>, 1x2 to [0] on 2, [0,1]>) {
  %0, %1 = "ifrt.Disassemble"(%arg0)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x4xi32>, 1x2 to [0] on 2, [0,1]>)
      -> (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 1, [0]>,
          !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 1, [1]>)
  return
}

// -----

func.func @disassemble_requires_outputs_on_single_devices(
    %arg0: !ifrt.array<tensor<2x4xi32>, 1x4 to [0, 1] on 2x2, [0,1,2,3]>) {
  // expected-error@+1 {{'ifrt.Disassemble' op requires every output to be a single device array. Actual: '!ifrt.array<tensor<2x2xi32>, 1x2 to [0] on 2, [0, 1]>'}}
  %0, %1 = "ifrt.Disassemble"(%arg0)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x4xi32>, 1x4 to [0, 1] on 2x2, [0,1,2,3]>)
      -> (!ifrt.array<tensor<2x2xi32>, 1x2 to [0] on 2, [0,1]>,
          !ifrt.array<tensor<2x2xi32>, 1x2 to [0] on 2, [2,3]>)
  return
}

// -----

func.func @disassemble_requires_same_device_list(
    %arg0: !ifrt.array<tensor<2x4xi32>, 1x2 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Disassemble' op requires the same input/output device list. Input 0, 1 vs Output 1, 2}}
  %0, %1 = "ifrt.Disassemble"(%arg0)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x4xi32>, 1x2 to [0] on 2, [0,1]>)
      -> (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 1, [1]>,
          !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 1, [2]>)
  return
}
