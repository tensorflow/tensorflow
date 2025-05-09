// RUN: emitters_opt %s -split-input-file -xla-gpu-lower-xla-shared | FileCheck %s

func.func @forall_op(%input: tensor<1024x32x2xf32>) -> (tensor<1024x32x2xf32>) {
  %double = xla.forall (%i, %j, %k) in (1024, 32, 2) with (%captured_input = %input) -> (tensor<1024x32x2xf32>) {
    %t = tensor.extract %captured_input[%i, %j, %k] : tensor<1024x32x2xf32>
    %add = arith.addf %t, %t : f32
    %result = tensor.insert %add into %captured_input[%i, %j, %k] : tensor<1024x32x2xf32>
    xla.yield %result : tensor<1024x32x2xf32>
  }
  func.return %double : tensor<1024x32x2xf32>
}
// CHECK: [[THREAD_X:.*]] = gpu.thread_id x {xla.range = [0 : index, 1023 : index]}
// CHECK: [[THREAD_Y:.*]] = gpu.thread_id y {xla.range = [0 : index, 31 : index]}
// CHECK: [[THREAD_Z:.*]] = gpu.thread_id z {xla.range = [0 : index, 1 : index]}


// -----

func.func @workgroup_id_op() -> (index, index, index) {
  %workgroup_id_x = xla.workgroup_id x {xla.range = [0 : index, 1023 : index]}
  %workgroup_id_y = xla.workgroup_id y
  %workgroup_id_z = xla.workgroup_id z
  func.return %workgroup_id_x, %workgroup_id_y, %workgroup_id_z : index, index, index
}
// CHECK: [[WORKGROUP_ID_X:.*]] = gpu.block_id x {xla.range = [0 : index, 1023 : index]}
// CHECK: [[WORKGROUP_ID_Y:.*]] = gpu.block_id y
// CHECK: [[WORKGROUP_ID_Z:.*]] = gpu.block_id z
