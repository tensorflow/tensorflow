// RUN: ifrt-opt %s --ifrt-legalize-to-vifrt --symbol-dce --mlir-print-op-generic -split-input-file | FileCheck %s
// RUN: ifrt-translate --serialize --ifrt_version=current --atom_program_version=current %s | ifrt-translate --deserialize | ifrt-opt > %t.0
// RUN: ifrt-opt %s > %t.1
// RUN: diff %t.0 %t.1

// ============ Types and attributes ============

// Verifies conversion of the array and control types, and devices and sharding
// param attributes.
!array_t0 = !ifrt.array<tensor<2x4xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array_t1 = !ifrt.array<tensor<2x4xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 2>, [2,3]>
// CHECK-LABEL: "type_array_and_control"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}):
func.func @type_array_and_control(%arg0: !array_t0) attributes {ifrt.function} {
  // CHECK: "vifrt.CopyArraysV1"(%[[ARG0]])
  // CHECK-SAME: <{
  // CHECK-DAG: donated = false
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [2, 3], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array_t0) -> !array_t1
  return
}

!array_us0 = !ifrt.array<tensor<2x4xi32>, #ifrt.sharding_unspecified, [0,1]>
!array_us1 = !ifrt.array<tensor<2x4xi32>, #ifrt.sharding_unspecified, [2,3]>
// CHECK-LABEL: "attr_unspecified_sharding"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}):
func.func @attr_unspecified_sharding(%arg0: !array_us0) attributes {ifrt.function} {
  // CHECK: "vifrt.CopyArraysV1"(%[[ARG0]])
  // CHECK-SAME: <{
  // CHECK-DAG: donated = false
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_unspecified_v1, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_unspecified_v1, [2, 3], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array_us0) -> !array_us1
  return
}


// Verify conversion of IntervalAttr, Mapping, ArrayMapping,
// MappingAttrArrayAttr and ArrayMappingAttrArrayAttr.
!array_rattr_in0 = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array_rattr_in1 = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!array_rattr_out = !ifrt.array<tensor<2x4xi32>,
                            #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
// CHECK-LABEL: "remap_attributes"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}):
func.func @remap_attributes(%arg0: !array_rattr_in0, %arg1: !array_rattr_in1)
    attributes {ifrt.function} {
  // CHECK: "vifrt.RemapArraysV1"(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: <{
  // CHECK-DAG: donated = false
  // CHECK-DAG: mappings = [#vifrt.array_mapping_v1<0, 0, [#vifrt.mapping_v1<[0 : 1 : 1] to [0 : 1 : 1]>]>, #vifrt.array_mapping_v1<1, 0, [#vifrt.mapping_v1<[0 : 1 : 1] to [1 : 2 : 1]>]>]
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 1>, [0], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 1>, [1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> !vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">
  %0 = ifrt.RemapArrays(%arg0, %arg1)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
      : (!array_rattr_in0, !array_rattr_in1) -> (!array_rattr_out)
  return
}

// CHECK-LABEL: "ifrt_function_attribute"
// CHECK-NOT: {ifrt.function}
// CHECK: {vifrt.function}
func.func @ifrt_function_attribute() attributes {ifrt.function} {
  return
}

// ============ Ops ============

!array_cp0 = !ifrt.array<tensor<2x4xi32>,
                         #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array_cp1 = !ifrt.array<tensor<2x4xi32>,
                         #ifrt.sharding_param<1x1 to [0] on 2>, [2,3]>
// CHECK-LABEL: "op_copy_arrays"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}):
func.func @op_copy_arrays(%arg0: !array_cp0) -> !array_cp1
    attributes {ifrt.function} {
  // CHECK: "vifrt.CopyArraysV1"(%[[ARG0]])
  // CHECK-SAME: <{
  // CHECK-DAG: donated = false
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [2, 3], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array_cp0) -> !array_cp1
  return %0: !array_cp1
}

!array_ad0 = !ifrt.array<tensor<2x2xi32>,
                         #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array_ad1 = !ifrt.array<tensor<2x2xi32>,
                         #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!array_ad2 = !ifrt.array<tensor<2x4xi32>,
                         #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
// CHECK-LABEL: "op_assemble"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}):
func.func @op_assemble(%arg0: !array_ad0, %arg1: !array_ad1)
    attributes {ifrt.function} {
  // CHECK: "vifrt.AssembleV1"(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: <{
  // CHECK-DAG: operandSegmentSizes = array<i32: 2, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 1>, [0], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 1>, [1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> !vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">
  %0 = "ifrt.Assemble"(%arg0, %arg1) {operandSegmentSizes=array<i32: 2, 0>}
      : (!array_ad0, !array_ad1) -> !array_ad2
  return
}

// CHECK-LABEL: "op_disassemble"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}):
func.func @op_disassemble(%arg0: !array_ad2) attributes {ifrt.function} {
  // CHECK: "vifrt.DisassembleV1"(%[[ARG0]])
  // CHECK-SAME: {
  // CHECK-DAG: operand_segment_sizes = array<i32: 2, 0>
  // CHECK-SAME: }
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 1>, [0], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 1>, [1], memory_kind = "vifrt.default", layout = "vifrt.default">)
  %0, %1 = "ifrt.Disassemble"(%arg0) {operand_segment_sizes=array<i32: 2, 0>}
      : (!array_ad2) -> (!array_ad0, !array_ad1)
  return
}

// CHECK-LABEL: "op_after"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}):
func.func @op_after(%arg0: !array_cp0, %arg1: !array_cp1)
    -> (!array_cp0, !array_cp0) attributes {ifrt.function} {
  // CHECK: %[[OUT:.+]]:2 = "vifrt.CopyArraysV1"(%[[ARG0]])
  // CHECK-SAME: <{
  // CHECK-DAG: donated = false
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [2, 3], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array_cp0) -> !array_cp1
  // CHECK: "vifrt.CopyArraysV1"(%[[OUT]]#0, %[[ARG1]], %[[OUT]]#1)
  // CHECK-SAME: <{
  // CHECK-DAG: donated = false
  // CHECK-DAG: operandSegmentSizes = array<i32: 2, 1>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [2, 3], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [2, 3], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1) -> (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %1, %2, %ctrl_1 = ifrt.CopyArrays(%0, %arg1) after %ctrl_0
      : (!array_cp1, !array_cp1) -> (!array_cp0, !array_cp0)
  return %1, %2: !array_cp0, !array_cp0
}

!array_le_in = !ifrt.array<tensor<2x2xi32>,
                           #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array_le_out = !ifrt.array<tensor<4x4xi32>,
                            #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
// CHECK-LABEL: "op_call_loaded_executable"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}):
func.func @op_call_loaded_executable(
    %arg0: !array_le_in {ifrt.donated}, %arg1: !array_le_in {ifrt.donated})
    attributes {ifrt.function} {
  // CHECK: %[[OUT:.+]]:2 = "vifrt.CallLoadedExecutableV1"(%[[ARG0]])
  // CHECK-SAME: <{
  // CHECK-DAG: callee = @test_loaded_executable1
  // CHECK-DAG: donated_input_indices = array<i32>
  // CHECK-DAG: io_aliases = [],
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<4x4xi32>, #vifrt.sharding_param_v1<1x2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @test_loaded_executable1(%arg0)
      : (!array_le_in) -> !array_le_out
  // CHECK: "vifrt.CallLoadedExecutableV1"(%[[ARG0]])
  // CHECK-SAME: <{
  // CHECK-DAG: callee = @test_loaded_executable1
  // CHECK-DAG: donated_input_indices = array<i32: 0>
  // CHECK-DAG: io_aliases = []
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<4x4xi32>, #vifrt.sharding_param_v1<1x2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %1, %ctrl_1 = ifrt.CallLoadedExecutable @test_loaded_executable1(%arg0)
      {donated_input_indices=array<i32: 0>} : (!array_le_in) -> !array_le_out
  // CHECK: "vifrt.CallLoadedExecutableV1"(%[[ARG1]])
  // CHECK-SAME: <{
  // CHECK-DAG: callee = @test_loaded_executable2
  // CHECK-DAG: donated_input_indices = array<i32>
  // CHECK-DAG: io_aliases = [array<i32: 0, 0>]
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %2, %ctrl_2 = ifrt.CallLoadedExecutable @test_loaded_executable2(%arg1)
      {io_aliases=[array<i32: 0, 0>]} : (!array_le_in) -> !array_le_in
  return
}

// CHECK: "vifrt.LoadedExecutableV1"()
// CHECK-SAME: <{
// CHECK-DAG: devices = #vifrt<devices_v1[0, 1]>
// CHECK-DAG: function_type = #vifrt.type_v1<!vifrt.func_v1<(!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> !vifrt.array_v1<tensor<4x4xi32>, #vifrt.sharding_param_v1<1x2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">>>
// CHECK-DAG: sym_name = "test_loaded_executable1"
// CHECK-SAME: }>
ifrt.LoadedExecutable @test_loaded_executable1 on devices [0,1]
   : (!array_le_in) -> !array_le_out

// CHECK: "vifrt.LoadedExecutableV1"()
// CHECK-SAME: <{
// CHECK-DAG: devices = #vifrt<devices_v1[0, 1]>
// CHECK-DAG: function_type = #vifrt.type_v1<!vifrt.func_v1<(!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> !vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">>>
// CHECK-DAG: sym_name = "test_loaded_executable2"
// CHECK-SAME: }>
ifrt.LoadedExecutable @test_loaded_executable2 on devices [0,1]
   : (!array_le_in) -> !array_le_in

!array_ra_in0 = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array_ra_in1 = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!array_ra_out = !ifrt.array<tensor<2x4xi32>,
                            #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
// CHECK-LABEL: "op_remap_arrays"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}):
func.func @op_remap_arrays(%arg0: !array_ra_in0, %arg1: !array_ra_in1)
    attributes {ifrt.function} {
  // CHECK: "vifrt.RemapArraysV1"(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: <{
  // CHECK-DAG: donated = false
  // CHECK-DAG: mappings = [#vifrt.array_mapping_v1<0, 0, [#vifrt.mapping_v1<[0 : 1 : 1] to [0 : 1 : 1]>]>, #vifrt.array_mapping_v1<1, 0, [#vifrt.mapping_v1<[0 : 1 : 1] to [1 : 2 : 1]>]>]
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 1>, [0], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 1>, [1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> !vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">
  %0 = ifrt.RemapArrays(%arg0, %arg1)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
      : (!array_ra_in0, !array_ra_in1) -> (!array_ra_out)
  return
}


!array_r0 = !ifrt.array<tensor<2xi32>,
                        #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
!array_r1 = !ifrt.array<tensor<2xi32>,
                        #ifrt.sharding_param<1 to [0] on 1>, [2]>
!array_r2 = !ifrt.array<tensor<2xi32>,
                       #ifrt.sharding_param<1 to [0] on 1>, [3]>
// CHECK-LABEL: "op_reshard"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}):
func.func @op_reshard(%arg0: !array_r0, %arg1: !array_r0)
    -> (!array_r1, !array_r2) attributes {ifrt.function} {
  // CHECK: "vifrt.ReshardV1"(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: <{
  // CHECK-DAG: donated = false
  // CHECK-DAG: operandSegmentSizes = array<i32: 2, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<1 to [0] on 1>, [2], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<1 to [0] on 1>, [3], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %0, %1, %ctrl_1 = ifrt.Reshard(%arg0, %arg1)
      : (!array_r0, !array_r0) -> (!array_r1, !array_r2)
  return %0, %1 : !array_r1, !array_r2
}

// Verifies that the FuncOp, ReturnOp and donated arguments are converted to
// VIFRT.

// CHECK: "vifrt.FuncV1"()
// CHECK-SAME: <{
// CHECK-DAG: arg_attrs = [{vifrt.donated}, {vifrt.donated}]
// CHECK-DAG: function_type = #vifrt.type_v1<!vifrt.func_v1<(!vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> !vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<1 to [0] on 1>, [2], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<1 to [0] on 1>, [3], memory_kind = "vifrt.default", layout = "vifrt.default">>>
// CHECK-DAG: sym_name = "donated_arguments"
// CHECK-DAG: res_attrs = []
// CHECK-SAME: }>
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}):
func.func @donated_arguments(
    %arg0: !array_r0 {ifrt.donated}, %arg1: !array_r0 {ifrt.donated})
    -> (!array_r1, !array_r2) attributes {ifrt.function} {
  // CHECK: %[[OUT:.+]]:3 = "vifrt.ReshardV1"(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: <{
  // CHECK-DAG: donated = true
  // CHECK-DAG: operandSegmentSizes = array<i32: 2, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<1 to [0] on 1>, [2], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<1 to [0] on 1>, [3], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %0, %1, %ctrl_1 = ifrt.Reshard(%arg0, %arg1) {donated=true}
      : (!array_r0, !array_r0) -> (!array_r1, !array_r2)
  // CHECK: "vifrt.ReturnV1"(%[[OUT]]#0, %[[OUT]]#1) : (!vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<1 to [0] on 1>, [2], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.array_v1<tensor<2xi32>, #vifrt.sharding_param_v1<1 to [0] on 1>, [3], memory_kind = "vifrt.default", layout = "vifrt.default">)
  return %0, %1 : !array_r1, !array_r2
}

// CHECK-LABEL: "op_func_call"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}):
func.func @op_func_call(%arg0: !array_cp0) -> !array_cp1
    attributes {ifrt.function} {
  // CHECK: %[[OUT0:.+]]:2 = "vifrt.CopyArraysV1"(%[[ARG0]])
  // CHECK-SAME: <{
  // CHECK-DAG: donated = false
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [2, 3], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array_cp0) -> !array_cp1
  // CHECK: %[[OUT1:.+]] = "vifrt.CallFuncV1"(%[[OUT0]]#0)
  // CHECK-SAME: <{callee = @copy_back}>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [2, 3], memory_kind = "vifrt.default", layout = "vifrt.default">) -> !vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">
  %1 = func.call @copy_back(%0) : (!array_cp1) -> !array_cp0
  return %0: !array_cp1
}

// CHECK: "vifrt.FuncV1"()
// CHECK-SAME: <{
// CHECK-DAG: arg_attrs = []
// CHECK-DAG: function_type = #vifrt.type_v1<!vifrt.func_v1<(!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [2, 3], memory_kind = "vifrt.default", layout = "vifrt.default">) -> !vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">>>
// CHECK-DAG: res_attrs = []
// CHECK-DAG: sym_name = "copy_back"
// CHECK-DAG: sym_visibility = "vifrt.default"
// CHECK-SAME: }>
// CHECK-NEXT: (%[[ARG1:.*]]: {{.*}}):
func.func @copy_back(%arg1: !array_cp1) -> !array_cp0
    attributes {ifrt.function} {
  // CHECK: "vifrt.CopyArraysV1"(%[[ARG1]])
  // CHECK-SAME: <{
  // CHECK-DAG: donated = false
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [2, 3], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v1<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %0, %ctrl = ifrt.CopyArrays(%arg1) : (!array_cp1) -> !array_cp0
  return %0: !array_cp0
}

// Important: The test verifying CallOps must be last. This is necessary because
// in order to test serialization roundtrip the tests in this file are not split
// into per file tests. However, during deserialization we do not know where to
// re-introduce the atom program modules within the module, and thus we append
// them at the end.
!array_op_call = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK-LABEL: "op_call"
// CHECK-NEXT: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}):
func.func @op_call(
    %arg0: !array_op_call {ifrt.donated}, %arg1: !array_op_call {ifrt.donated})
    -> !array_op_call attributes {ifrt.function} {
  // CHECK: %[[OUT0:.+]]:2 = "vifrt.CallV1"(%[[ARG0]])
  // CHECK-SAME: <{
  // CHECK-DAG: callee = "@add_one::@main"
  // CHECK-DAG: devices = #vifrt<devices_v1[0, 1]>
  // CHECK-DAG: donated_input_indices = array<i32>
  // CHECK-DAG: io_aliases = []
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<2x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<2x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
      : (!array_op_call) -> !array_op_call

  // Verifies that the control value is passed to the next call.

  // CHECK: %[[OUT1:.+]]:2 = "vifrt.CallV1"(%[[OUT0]]#0, %[[OUT0]]#1)
  // CHECK-SAME: <{
  // CHECK-DAG: callee = "@add_one::@main"
  // CHECK-DAG: devices = #vifrt<devices_v1[0, 1]>
  // CHECK-DAG: donated_input_indices = array<i32>
  // CHECK-DAG: io_aliases = []
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<2x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1) -> (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<2x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %1, %ctrl_1 = ifrt.Call @add_one::@main(%0) after %ctrl_0 on devices [0,1]
      : (!array_op_call) -> !array_op_call

  // Verifies that escaped symbol attr is correctly handled.
  // CHECK: %[[OUT2:.+]]:2 = "vifrt.CallV1"(%[[ARG0]])
  // CHECK-SAME: <{
  // CHECK-DAG: callee = "@escaped-module::@main"
  // CHECK-DAG: devices = #vifrt<devices_v1[0, 1]>
  // CHECK-DAG: donated_input_indices = array<i32>
  // CHECK-DAG: io_aliases = []
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<2x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<2x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %2, %ctrl_2 = ifrt.Call @"escaped-module"::@main(%arg0) on devices [0,1]
      : (!array_op_call) -> !array_op_call

  // Verifies that the donated input indices attribute is converted.

  // CHECK: "vifrt.CallV1"(%[[ARG0]])
  // CHECK-SAME: <{
  // CHECK-DAG: callee = "@add_one::@main"
  // CHECK-DAG: devices = #vifrt<devices_v1[0, 1]>
  // CHECK-DAG: donated_input_indices = array<i32: 0>
  // CHECK-DAG: io_aliases = []
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<2x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<2x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %3, %ctrl_3 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
      {donated_input_indices=array<i32: 0>} : (!array_op_call) -> !array_op_call

  // Verifies that the io_aliases attribute is converted.

  // CHECK: "vifrt.CallV1"(%[[ARG1]])
  // CHECK-SAME: <{
  // CHECK-DAG: callee = "@add_two::@main"
  // CHECK-DAG: devices = #vifrt<devices_v1[0, 1]>
  // CHECK-DAG: donated_input_indices = array<i32>,
  // CHECK-DAG: io_aliases = [array<i32: 0, 0>]
  // CHECK-DAG: operandSegmentSizes = array<i32: 1, 0>
  // CHECK-SAME: }>
  // CHECK-SAME: (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<2x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">) -> (!vifrt.array_v1<tensor<2x2xi32>, #vifrt.sharding_param_v1<2x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">, !vifrt.control_v1)
  %4, %ctrl_4 = ifrt.Call @add_two::@main(%arg1) on devices [0,1]
      {io_aliases=[array<i32: 0, 0>]} : (!array_op_call) -> !array_op_call

  return %1 : !array_op_call
}

// CHECK-NOT @add_one
module @add_one attributes {sym_visibility = "private"} {
  func.func private @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// CHECK-NOT @"escaped-module"
module @"escaped-module" attributes {sym_visibility = "private"} {
  func.func private @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<2> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// CHECK-NOT @add_two
module @add_two attributes {sym_visibility = "private"} {
  func.func private @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<2> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
