// RUN: emitters_opt %s --xla-cpu-lower-to-llvm -split-input-file | FileCheck %s

func.func @fn(%arg0: index, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  return %arg1 : tensor<2xi32>
}

func.func @kernel_prototype(%arg0: !xla_cpu.call_frame) -> !xla_cpu.error {
  %thread_id = xla_cpu.thread_id x in %arg0
  %0 = xla_cpu.load %arg0, 0 : tensor<2xi32>
  // Call a function so that its arguments are not optimized away.
  %1 = call @fn(%thread_id, %0) : (index, tensor<2xi32>) -> tensor<2xi32>
  xla_cpu.store %1 into %arg0, 1 : tensor<2xi32>
  %error = xla_cpu.success : !xla_cpu.error
  return %error : !xla_cpu.error
}

// CHECK-LABEL: @fn
// CHECK-NEXT:    return
// CHECK-LABEL: @kernel_prototype
// CHECK:         %[[CALL_FRAME_PTR:.+]]: !llvm.ptr
// CHECK:       ) -> !llvm.ptr {
// CHECK-DAG:     %[[ERROR:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:     %[[TID_GEP:.+]] = llvm.getelementptr inbounds %[[CALL_FRAME_PTR]][1]
// CHECK-DAG:     %[[TID_PTR:.+]] = llvm.load %[[TID_GEP]]
// CHECK-DAG:     %[[TID:.+]] = llvm.load %[[TID_PTR]]
// CHECK-DAG:     %[[TID_IDX:.+]] = {{.*}}cast %[[TID]] : i64 to index
// CHECK-DAG:     %[[KERNEL_ARG_GEP:.+]] = llvm.getelementptr inbounds %[[CALL_FRAME_PTR]][0, 3]
// CHECK-DAG:     %[[KERNEL_ARG_PTR:.+]] = llvm.load %[[KERNEL_ARG_GEP]]
// CHECK-DAG:     %[[ARG_GEP:.+]] = llvm.getelementptr inbounds %[[KERNEL_ARG_PTR]][0, 0]
// CHECK-DAG:     %[[ARG:.+]] = llvm.load %[[ARG_GEP]]
// CHECK-DAG:     %[[ARG_TENSOR:.+]] = {{.*}}cast %[[ARG]] : !llvm.ptr to tensor<2xi32>
// CHECK:         call @fn(%[[TID_IDX]], %[[ARG_TENSOR]])
// CHECK:         return %[[ERROR]] : !llvm.ptr
// CHECK: }

// -----

func.func @several_inputs(%arg0: index, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  return %arg1 : tensor<2xi32>
}

// CHECK-NOT: !llvm.ptr

// -----

func.func @several_results(%arg0: tensor<2xi32>)
    -> (tensor<2xi32>, tensor<2xi32>) {
  return %arg0, %arg0 : tensor<2xi32>, tensor<2xi32>
}

// CHECK-NOT: !llvm.ptr

// -----

func.func @input_not_call_frame(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  return %arg0 : tensor<2xi32>
}

// CHECK-NOT: !llvm.ptr

// -----

func.func @output_not_error(%arg0: !xla_cpu.call_frame) -> index {
  %thread_id = xla_cpu.thread_id x in %arg0
  return %thread_id : index
}

// CHECK: func.func @output_not_error(%arg0: !xla_cpu.call_frame) -> index {

// -----

func.func @get_z_thread_id(%arg0: !xla_cpu.call_frame) -> index {
  %thread_id = xla_cpu.thread_id z in %arg0
  return %thread_id : index
}

// CHECK [[TID_GEP:%.+]] = llvm.getelementptr
// CHECK [[TID_PTR:%.+]] = llvm.load [[TID_GEP]]
// CHECK [[TID_Z_GEP:%.+]] = llvm.getelementptr inbounds [[TID_PTR]][2]
