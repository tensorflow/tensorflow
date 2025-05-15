// RUN: emitters_opt %s -split-input-file -xla-gpu-lower-xla-shared \
// RUN: | FileCheck %s

func.func @forall_op(%input: tensor<1024x32x2xf32>) -> (tensor<1024x32x2xf32>) {
  %double = scf.forall (%i, %j, %k) in (1024, 32, 2)
      shared_outs(%captured_input = %input) -> (tensor<1024x32x2xf32>) {
    %t = tensor.extract %captured_input[%i, %j, %k] : tensor<1024x32x2xf32>
    %add = arith.addf %t, %t : f32
    %result = tensor.insert %add into %captured_input[%i, %j, %k]
        : tensor<1024x32x2xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %result into %captured_input[0, 0, 0][1024, 32, 2][1, 1, 1]
          : tensor<1024x32x2xf32> into tensor<1024x32x2xf32>
    }
  }
  func.return %double : tensor<1024x32x2xf32>
}
// CHECK: [[THREAD_X:.*]] = gpu.thread_id x {xla.range = [0 : index, 1023 : index]}
// CHECK: [[THREAD_Y:.*]] = gpu.thread_id y {xla.range = [0 : index, 31 : index]}
// CHECK: [[THREAD_Z:.*]] = gpu.thread_id z {xla.range = [0 : index, 1 : index]}
