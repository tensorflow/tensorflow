// RUN: emitters_opt %s -split-input-file -xla-cpu-lower-xla-shared | FileCheck %s

func.func @forall_op(%input: tensor<1024x32x2xf32>) -> (tensor<1024x32x2xf32>) {
  %double = xla.forall (%i, %j, %k) in (1024, 32, 2) with (%captured_input = %input) -> (tensor<1024x32x2xf32>) {
    %t = tensor.extract %captured_input[%i, %j, %k] : tensor<1024x32x2xf32>
    %add = arith.addf %t, %t : f32
    %result = tensor.insert %add into %captured_input[%i, %j, %k] : tensor<1024x32x2xf32>
    xla.yield %result : tensor<1024x32x2xf32>
  }
  func.return %double : tensor<1024x32x2xf32>
}
// CHECK-DAG: [[CONST_0:%.*]] = arith.constant 0 : index
// CHECK-DAG: [[CONST_1:%.*]] = arith.constant 1 : index
// CHECK-DAG: [[CONST_1024:%.*]] = arith.constant 1024 : index
// CHECK-DAG: [[CONST_32:%.*]] = arith.constant 32 : index
// CHECK-DAG: [[CONST_2:%.*]] = arith.constant 2 : index
// CHECK: [[FOR_ALL_0:%.*]] = scf.for {{.*}} = [[CONST_0]] to [[CONST_1024]] step [[CONST_1]]
// CHECK: [[FOR_ALL_1:%.*]] = scf.for {{.*}} = [[CONST_0]] to [[CONST_32]] step [[CONST_1]]
// CHECK: [[FOR_ALL_2:%.*]] = scf.for {{.*}} = [[CONST_0]] to [[CONST_2]] step [[CONST_1]]

// -----

func.func @workgroup_id_op(%arg0: !xla_cpu.call_frame) -> (index, index, index) {
  %workgroup_id_x = xla.workgroup_id x
  %workgroup_id_y = xla.workgroup_id y
  %workgroup_id_z = xla.workgroup_id z
  func.return %workgroup_id_x, %workgroup_id_y, %workgroup_id_z : index, index, index
}
// CHECK: func.func @workgroup_id_op([[CALL_FRAME:%.+]]:
// CHECK: [[THREAD_ID_X:.*]] = xla_cpu.thread_id x in [[CALL_FRAME]]
// CHECK: [[THREAD_ID_Y:.*]] = xla_cpu.thread_id y in [[CALL_FRAME]]
// CHECK: [[THREAD_ID_Z:.*]] = xla_cpu.thread_id z in [[CALL_FRAME]]
