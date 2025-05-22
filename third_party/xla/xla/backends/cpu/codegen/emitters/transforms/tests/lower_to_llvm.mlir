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

func.func private @wrap_entry(%arg0: tensor<2xi32>, %arg1: tensor<21x12xi32>)
    -> (tensor<2xi32>, tensor<21x12xi32>, index, index, index)
    attributes {xla.entry}
{
  %workgroup_x = xla.workgroup_id x
  %workgroup_y = xla.workgroup_id y
  %workgroup_z = xla.workgroup_id z
  return %arg0, %arg1, %workgroup_y, %workgroup_x, %workgroup_z
      : tensor<2xi32>, tensor<21x12xi32>, index, index, index
}

// CHECK:  func.func @wrap_entry_kernel(%[[CALL_FRAME:.+]]: !llvm.ptr) -> !llvm.ptr attributes
// CHECK-SAME: frame_pointer = #llvm.framePointerKind<all>
// CHECK-SAME: "prefer-vector-width", "128"
// CHECK-SAME: uwtable_kind = #llvm.uwtableKind<async>
// CHECK-DAG:    %[[RETURN_PTR:.+]] = llvm.mlir.zero
// CHECK-DAG:    %[[TENSOR_GEP:.+]] = llvm.getelementptr inbounds %[[CALL_FRAME]][0, 3]
// CHECK-DAG:    %[[TENSOR_PTR:.+]] = llvm.load %[[TENSOR_GEP]] invariant
// CHECK-DAG:    %[[TENSOR_0_GEP:.+]] = llvm.getelementptr inbounds %[[TENSOR_PTR]][0, 0]
// CHECK-DAG:    %[[TENSOR_0_PTR:.+]] = llvm.load %[[TENSOR_0_GEP]] invariant {llvm.align = 32 : index}
// CHECK-DAG:    %[[TENSOR_0:.+]] = builtin.unrealized_conversion_cast %[[TENSOR_0_PTR]]
// CHECK-DAG:    %[[TENSOR_1_GEP:.+]] = llvm.getelementptr inbounds %[[TENSOR_PTR]][1, 0]
// CHECK-DAG:    %[[TENSOR_1_PTR:.+]] = llvm.load %[[TENSOR_1_GEP]] invariant {llvm.align = 32 : index}
// CHECK-DAG:    %[[TENSOR_1:.+]] = builtin.unrealized_conversion_cast %[[TENSOR_1_PTR]]
// CHECK-DAG:    %[[WORKGROUP_IDS_GEP:.+]] = llvm.getelementptr inbounds %[[CALL_FRAME]][0, 1]
// CHECK-DAG:    %[[WORK_IDS_PTR:.+]] = llvm.load %[[WORKGROUP_IDS_GEP]]
// CHECK-DAG:    %[[WORK_ID_0_GEP:.+]] = llvm.getelementptr inbounds %[[WORK_IDS_PTR]][0, 0]
// CHECK-DAG:    %[[WORK_ID_0_PTR:.+]] = llvm.load %[[WORK_ID_0_GEP]]
// CHECK-DAG:    %[[WORK_ID_0:.+]] = builtin.unrealized_conversion_cast %[[WORK_ID_0_PTR]]
// CHECK-DAG:    %[[WORK_ID_1_GEP:.+]] = llvm.getelementptr inbounds %[[WORK_IDS_PTR]][0, 1]
// CHECK-DAG:    %[[WORK_ID_1_PTR:.+]] = llvm.load %[[WORK_ID_1_GEP]]
// CHECK-DAG:    %[[WORK_ID_1:.+]] = builtin.unrealized_conversion_cast %[[WORK_ID_1_PTR]]
// CHECK-DAG:    %[[WORK_ID_2_GEP:.+]] = llvm.getelementptr inbounds %[[WORK_IDS_PTR]][0, 2]
// CHECK-DAG:    %[[WORK_ID_2_PTR:.+]] = llvm.load %[[WORK_ID_2_GEP]]
// CHECK-DAG:    %[[WORK_ID_2:.+]] = builtin.unrealized_conversion_cast %[[WORK_ID_2_PTR]]
// CHECK-DAG:    call @wrap_entry(%[[TENSOR_0]],
// CHECK-DAG                              %[[TENSOR_1]],
// CHECK-DAG                              %[[WORK_ID_0]],
// CHECK-DAG                              %[[WORK_ID_1]],
// CHECK-DAG                              %[[WORK_ID_2]])
// CHECK-DAG:    return %[[RETURN_PTR]]
// CHECK:      }
// CHECK:      func.func private @wrap_entry(
// CHECK:         %[[ARG_0:.+]]: tensor<2xi32>, %[[ARG_1:.+]]: tensor<21x12xi32>,
// CHECK:         %[[ARG_2:.+]]: index, %[[ARG_3:.+]]: index, %[[ARG_4:.+]]: index)
// CHECK:       return %[[ARG_0]], %[[ARG_1]], %[[ARG_3]], %[[ARG_2]], %[[ARG_4]]
// CHECK:      }
