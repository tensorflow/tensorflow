// RUN: emitters_opt %s --xla-cpu-lower-to-llvm="prefer_vector_width=128" -split-input-file -cse | FileCheck %s

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
  %thread_id = xla.workgroup_id x
  return %thread_id : index
}

// CHECK: func.func @output_not_error(%arg0: !xla_cpu.call_frame) -> index {

// -----

func.func @extract_workgroup_id(%call_frame: !xla_cpu.call_frame) -> (index, index, index) {
  %id_x = xla_cpu.extract_workgroup_id %call_frame, x
  %id_y = xla_cpu.extract_workgroup_id %call_frame, y
  %id_z = xla_cpu.extract_workgroup_id %call_frame, z
  return %id_x, %id_y, %id_z : index, index, index
}

// CHECK-LABEL: func.func @extract_workgroup_id(
// CHECK-SAME: %[[CALL_FRAME:.+]]: !xla_cpu.call_frame) -> (index, index, index) {
// CHECK: %[[CALL_FRAME_PTR:.+]] = builtin.unrealized_conversion_cast %[[CALL_FRAME]] : !xla_cpu.call_frame to !llvm.ptr
// CHECK: %[[WORKGROUP_GEP:.+]] = llvm.getelementptr inbounds %[[CALL_FRAME_PTR]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
// CHECK: %[[WORKGROUP_PTR:.+]] = llvm.load %[[WORKGROUP_GEP]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[WORKGROUP_X_GEP:.+]] = llvm.getelementptr inbounds %[[WORKGROUP_PTR]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
// CHECK: %[[WORKGROUP_X:.+]] = llvm.load %[[WORKGROUP_X_GEP]] invariant : !llvm.ptr -> i64
// CHECK: %[[WORKGROUP_X_IDX:.+]] = builtin.unrealized_conversion_cast %[[WORKGROUP_X]] : i64 to index
// CHECK: %[[WORKGROUP_Y_GEP:.+]] = llvm.getelementptr inbounds %[[WORKGROUP_PTR]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
// CHECK: %[[WORKGROUP_Y:.+]] = llvm.load %[[WORKGROUP_Y_GEP]] invariant : !llvm.ptr -> i64
// CHECK: %[[WORKGROUP_Y_IDX:.+]] = builtin.unrealized_conversion_cast %[[WORKGROUP_Y]] : i64 to index
// CHECK: %[[WORKGROUP_Z_GEP:.+]] = llvm.getelementptr inbounds %[[WORKGROUP_PTR]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
// CHECK: %[[WORKGROUP_Z:.+]] = llvm.load %[[WORKGROUP_Z_GEP]] invariant : !llvm.ptr -> i64
// CHECK: %[[WORKGROUP_Z_IDX:.+]] = builtin.unrealized_conversion_cast %[[WORKGROUP_Z]] : i64 to index
// CHECK: return %[[WORKGROUP_X_IDX]], %[[WORKGROUP_Y_IDX]], %[[WORKGROUP_Z_IDX]] : index, index, index

// -----

func.func @load_memref(%call_frame: !xla_cpu.call_frame) -> memref<2x4xi32> {
  %loaded = xla_cpu.load %call_frame, 0 : memref<2x4xi32>
  return %loaded : memref<2x4xi32>
}

// CHECK-LABEL: @load_memref(
// CHECK-SAME:    %[[CALLFRAME:.*]]: !xla_cpu.call_frame) -> memref<2x4xi32> {
// CHECK-DAG:  %[[C_1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:  %[[C_4:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-DAG:  %[[C_2:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:  %[[C_0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:  %[[INIT:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:  %[[CALLFRAME_PTR:.*]] = builtin.unrealized_conversion_cast %[[CALLFRAME]] : !xla_cpu.call_frame to !llvm.ptr
// CHECK-DAG:  %[[ARGS_GEP:.*]] = llvm.getelementptr inbounds %[[CALLFRAME_PTR]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
// CHECK-DAG:  %[[ARGS_PTR:.*]] = llvm.load %[[ARGS_GEP]] invariant : !llvm.ptr -> !llvm.ptr
// CHECK-DAG:  %[[INPUT_GEP:.*]] = llvm.getelementptr inbounds %[[ARGS_PTR]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
// CHECK-DAG:  %[[INPUT_PTR:.*]] = llvm.load %[[INPUT_GEP]] invariant : !llvm.ptr -> !llvm.ptr
// CHECK-DAG:  %[[INSERT_0:.*]] = llvm.insertvalue %[[INPUT_PTR]], %[[INIT]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:  %[[INSERT_1:.*]] = llvm.insertvalue %[[INPUT_PTR]], %[[INSERT_0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:  %[[INSERT_2:.*]] = llvm.insertvalue %[[C_0]], %[[INSERT_1]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:  %[[INSERT_3:.*]] = llvm.insertvalue %[[C_2]], %[[INSERT_2]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:  %[[INSERT_4:.*]] = llvm.insertvalue %[[C_4]], %[[INSERT_3]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:  %[[INSERT_5:.*]] = llvm.insertvalue %[[C_4]], %[[INSERT_4]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:  %[[FINAL_DESC:.*]] = llvm.insertvalue %[[C_1]], %[[INSERT_5]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:  %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[FINAL_DESC]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<2x4xi32>
// CHECK:  return %[[MEMREF]] : memref<2x4xi32>


// -----

func.func private @wrap_entry(
  %arg0: tensor<2xi32> {llvm.dereferenceable = 8 : index},
  %arg1: tensor<21x12xi32> {llvm.dereferenceable = 1008 : index})
    -> (tensor<2xi32>, tensor<21x12xi32>, index, index, index)
    attributes {xla.entry}
{
  %workgroup_x = xla.workgroup_id x
  %workgroup_y = xla.workgroup_id y
  %workgroup_z = xla.workgroup_id z
  return %arg0, %arg1, %workgroup_y, %workgroup_x, %workgroup_z
      : tensor<2xi32>, tensor<21x12xi32>, index, index, index
}

// CHECK:  func.func @wrap_entry(%[[CALL_FRAME:.+]]: !llvm.ptr) -> !llvm.ptr attributes
// CHECK-SAME: frame_pointer = #llvm.framePointerKind<all>
// CHECK-SAME: "prefer-vector-width", "128"
// CHECK-SAME: uwtable_kind = #llvm.uwtableKind<async>
// CHECK-DAG:    %[[RETURN_PTR:.+]] = llvm.mlir.zero
// CHECK-DAG:    %[[TENSOR_GEP:.+]] = llvm.getelementptr inbounds %[[CALL_FRAME]][0, 3]
// CHECK-DAG:    %[[TENSOR_PTR:.+]] = llvm.load %[[TENSOR_GEP]] invariant
// CHECK-DAG:    %[[TENSOR_0_GEP:.+]] = llvm.getelementptr inbounds %[[TENSOR_PTR]][0, 0]
// CHECK-DAG:    %[[TENSOR_0_PTR:.+]] = llvm.load %[[TENSOR_0_GEP]] invariant dereferenceable<bytes = 8>
// CHECK-DAG:    %[[TENSOR_0:.+]] = builtin.unrealized_conversion_cast %[[TENSOR_0_PTR]]
// CHECK-DAG:    %[[TENSOR_1_GEP:.+]] = llvm.getelementptr inbounds %[[TENSOR_PTR]][1, 0]
// CHECK-DAG:    %[[TENSOR_1_PTR:.+]] = llvm.load %[[TENSOR_1_GEP]] invariant dereferenceable<bytes = 1008>
// CHECK-DAG:    %[[TENSOR_1:.+]] = builtin.unrealized_conversion_cast %[[TENSOR_1_PTR]]
// CHECK-DAG:    %[[WORKGROUP_IDS_GEP:.+]] = llvm.getelementptr inbounds %[[CALL_FRAME]][0, 1]
// CHECK-DAG:    %[[WORK_IDS_PTR:.+]] = llvm.load %[[WORKGROUP_IDS_GEP]]
// CHECK-DAG:    %[[WORK_ID_0_GEP:.+]] = llvm.getelementptr inbounds %[[WORK_IDS_PTR]][0, 0]
// CHECK-DAG:    %[[WORK_ID_0_PTR:.+]] = llvm.load %[[WORK_ID_0_GEP]] invariant
// CHECK-DAG:    %[[WORK_ID_0:.+]] = builtin.unrealized_conversion_cast %[[WORK_ID_0_PTR]]
// CHECK-DAG:    %[[WORK_ID_1_GEP:.+]] = llvm.getelementptr inbounds %[[WORK_IDS_PTR]][0, 1]
// CHECK-DAG:    %[[WORK_ID_1_PTR:.+]] = llvm.load %[[WORK_ID_1_GEP]] invariant
// CHECK-DAG:    %[[WORK_ID_1:.+]] = builtin.unrealized_conversion_cast %[[WORK_ID_1_PTR]]
// CHECK-DAG:    %[[WORK_ID_2_GEP:.+]] = llvm.getelementptr inbounds %[[WORK_IDS_PTR]][0, 2]
// CHECK-DAG:    %[[WORK_ID_2_PTR:.+]] = llvm.load %[[WORK_ID_2_GEP]] invariant
// CHECK-DAG:    %[[WORK_ID_2:.+]] = builtin.unrealized_conversion_cast %[[WORK_ID_2_PTR]]
// CHECK-DAG:    call @wrap_entry_wrapped(%[[TENSOR_0]],
// CHECK-DAG                              %[[TENSOR_1]],
// CHECK-DAG                              %[[WORK_ID_0]],
// CHECK-DAG                              %[[WORK_ID_1]],
// CHECK-DAG                              %[[WORK_ID_2]])
// CHECK-DAG:    return %[[RETURN_PTR]]
// CHECK:      }
// CHECK:      func.func private @wrap_entry_wrapped(
// CHECK:         %[[ARG_0:.+]]: tensor<2xi32> {llvm.dereferenceable = 8 : index},
// CHECK:         %[[ARG_1:.+]]: tensor<21x12xi32> {llvm.dereferenceable = 1008 : index},
// CHECK:         %[[ARG_2:.+]]: index, %[[ARG_3:.+]]: index, %[[ARG_4:.+]]: index)
// CHECK:       return %[[ARG_0]], %[[ARG_1]], %[[ARG_3]], %[[ARG_2]], %[[ARG_4]]
// CHECK:      }

// -----

func.func @test_8x8_vector_transpose_lowering(%arg0: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<8x8xf32> to vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// CHECK @test_8x8_vector_transpose_lowering(%[[ARG_0:.+]]: vector<8x8xf32>) -> vector<8x8xf32> {
// CHECK:      %[[POISON_RESULT:.+]] = ub.poison : vector<8x8xf32>
// CHECK:      %[[R0:.+]] = vector.extract %[[ARG_0]][0]
// CHECK:      %[[R1:.+]] = vector.extract %[[ARG_0]][1
// CHECK:      %[[R2:.+]] = vector.extract %[[ARG_0]][2]
// CHECK:      %[[R3:.+]] = vector.extract %[[ARG_0]][3]
// CHECK:      %[[R4:.+]] = vector.extract %[[ARG_0]][4]
// CHECK:      %[[R5:.+]] = vector.extract %[[ARG_0]][5]
// CHECK:      %[[R6:.+]] = vector.extract %[[ARG_0]][6]
// CHECK:      %[[R7:.+]] = vector.extract %[[ARG_0]][7]

// CHECK:      %[[T0:.+]] = vector.shuffle %[[R0]], %[[R1]] [0, 8, 2, 10, 4, 12, 6, 14] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[T1:.+]] = vector.shuffle %[[R0]], %[[R1]] [1, 9, 3, 11, 5, 13, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[T2:.+]] = vector.shuffle %[[R2]], %[[R3]] [0, 8, 2, 10, 4, 12, 6, 14] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[T3:.+]] = vector.shuffle %[[R2]], %[[R3]] [1, 9, 3, 11, 5, 13, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[T4:.+]] = vector.shuffle %[[R4]], %[[R5]] [0, 8, 2, 10, 4, 12, 6, 14] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[T5:.+]] = vector.shuffle %[[R4]], %[[R5]] [1, 9, 3, 11, 5, 13, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[T6:.+]] = vector.shuffle %[[R6]], %[[R7]] [0, 8, 2, 10, 4, 12, 6, 14] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[T7:.+]] = vector.shuffle %[[R6]], %[[R7]] [1, 9, 3, 11, 5, 13, 7, 15] : vector<8xf32>, vector<8xf32>

// CHECK:      %[[U0:.+]] = vector.shuffle %[[T0]], %[[T2]] [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[U2:.+]] = vector.shuffle %[[T0]], %[[T2]] [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[U1:.+]] = vector.shuffle %[[T1]], %[[T3]] [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[U3:.+]] = vector.shuffle %[[T1]], %[[T3]] [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[U4:.+]] = vector.shuffle %[[T4]], %[[T6]] [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[U6:.+]] = vector.shuffle %[[T4]], %[[T6]] [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[U5:.+]] = vector.shuffle %[[T5]], %[[T7]] [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[U7:.+]] = vector.shuffle %[[T5]], %[[T7]] [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>

// CHECK:      %[[W0:.+]] = vector.shuffle %[[U0]], %[[U4]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[W4:.+]] = vector.shuffle %[[U0]], %[[U4]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[W1:.+]] = vector.shuffle %[[U1]], %[[U5]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[W5:.+]] = vector.shuffle %[[U1]], %[[U5]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[W2:.+]] = vector.shuffle %[[U2]], %[[U6]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[W6:.+]] = vector.shuffle %[[U2]], %[[U6]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[W3:.+]] = vector.shuffle %[[U3]], %[[U7]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK:      %[[W7:.+]] = vector.shuffle %[[U3]], %[[U7]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>

// CHECK:      %[[RES_0:.+]] = vector.insert %[[W0]], %[[POISON_RESULT]] [0]
// CHECK:      %[[RES_1:.+]] = vector.insert %[[W1]], %[[RES_0]] [1]
// CHECK:      %[[RES_2:.+]] = vector.insert %[[W2]], %[[RES_1]] [2]
// CHECK:      %[[RES_3:.+]] = vector.insert %[[W3]], %[[RES_2]] [3]
// CHECK:      %[[RES_4:.+]] = vector.insert %[[W4]], %[[RES_3]] [4]
// CHECK:      %[[RES_5:.+]] = vector.insert %[[W5]], %[[RES_4]] [5]
// CHECK:      %[[RES_6:.+]] = vector.insert %[[W6]], %[[RES_5]] [6]
// CHECK:      %[[RES_7:.+]] = vector.insert %[[W7]], %[[RES_6]] [7]

// CHECK-NEXT: return %[[RES_7]] : vector<8x8xf32>
// CHECK:      }

// -----

func.func @test_other_vector_transpose_shape_falls_back_to_vector(%arg0: vector<8x16xf32>) -> vector<16x8xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<8x16xf32> to vector<16x8xf32>
  return %0 : vector<16x8xf32>
}

// CHECK @test_other_vector_transpose_shape_falls_back_to_vector(%[[ARG_0:.+]]: vector<8x16xf32>) -> vector<16x8xf32> {
// CHECK:      %[[RES:.+]] = vector.transpose %[[ARG_0]], [1, 0] : vector<8x16xf32> to vector<16x8xf32>
// CHECK-NEXT: return %[[RES]] : vector<16x8xf32>
// CHECK:      }
