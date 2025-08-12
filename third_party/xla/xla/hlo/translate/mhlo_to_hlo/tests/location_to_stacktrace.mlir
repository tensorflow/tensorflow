// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo %s | FileCheck %s

// Checks no locations

// CHECK-LABEL: hlo_module       {
// CHECK: name: "after-all.2"
// CHECK-NEXT: opcode: "after-all"
// CHECK-NEXT: shape {
// CHECK-NEXT: element_type: TOKEN
// CHECK-NEXT: }
// CHECK-NEXT: metadata {
// CHECK-NEXT: }
// CHECK: stack_frame_index {
// CHECK-NEXT: }
#loc = loc(unknown)
module @main attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: !mhlo.token loc(unknown)) -> !mhlo.token {
    %0 = mhlo.after_all %arg0 {xla_shape = "token[]"} : !mhlo.token loc(#loc)
    return %0 : !mhlo.token loc(#loc)
  } loc(#loc)
} loc(#loc)

// -----

// Checks single named frame location

// CHECK-LABEL: hlo_module       {

// CHECK: name: "after-all.2"
// CHECK-NEXT: opcode: "after-all"
// CHECK-NEXT: shape {
// CHECK-NEXT:   element_type: TOKEN
// CHECK-NEXT: }
// CHECK-NEXT: metadata {
// CHECK-NEXT:   op_name: "name(anothername)
// CHECK-NEXT:   source_file: "file_name"
// CHECK-NEXT:   source_line: 2
// CHECK-NEXT:   stack_frame_id: 1
// CHECK-NEXT: }

// CHECK: stack_frame_index {
// CHECK-NEXT: file_names: "file_name"
// CHECK-NEXT: function_names: "function_name"
// CHECK-NEXT: file_locations {
// CHECK-NEXT:   file_name_id: 1
// CHECK-NEXT:   function_name_id: 1
// CHECK-NEXT:   line: 2
// CHECK-NEXT:   column: 8
// CHECK-NEXT: }
// CHECK-NEXT: stack_frames {
// CHECK-NEXT:   file_location_id: 1
// CHECK-NEXT: }
// CHECK-NEXT: }
#loc = loc(unknown)
module @main attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: !mhlo.token loc(unknown)) -> !mhlo.token {
    %0 = mhlo.after_all %arg0 {xla_shape = "token[]"} : !mhlo.token loc(#op_loc)
    return %0 : !mhlo.token loc(#loc)
  } loc(#loc)
} loc(#loc)
#frame_file_loc = loc("file_name":2:8)
#frame_loc = loc("function_name"(#frame_file_loc))
#op_loc = loc("name(anothername)"(#frame_loc))

// -----

// CHECK-LABEL: hlo_module       {

// CHECK: name: "after-all.2"
// CHECK-NEXT: opcode: "after-all"
// CHECK-NEXT: shape {
// CHECK-NEXT:   element_type: TOKEN
// CHECK-NEXT: }
// CHECK-NEXT: metadata {
// CHECK-NEXT:   op_type: "atype"
// CHECK-NEXT:   op_name: "name(anothername)
// CHECK-NEXT:   source_file: "file_name_2"
// CHECK-NEXT:   source_line: 3
// CHECK-NEXT:   stack_frame_id: 2
// CHECK-NEXT: }

// CHECK: stack_frame_index {
// CHECK-NEXT: file_names: "file_name"
// CHECK-NEXT: file_names: "file_name_2"
// CHECK-NEXT: function_names: "function_name"
// CHECK-NEXT: function_names: "function_name_2"
// CHECK-NEXT: file_locations {
// CHECK-NEXT:  file_name_id: 1
// CHECK-NEXT:  function_name_id: 1
// CHECK-NEXT:  line: 2
// CHECK-NEXT:  column: 8
// CHECK-NEXT: }
// CHECK-NEXT: file_locations {
// CHECK-NEXT:  file_name_id: 2
// CHECK-NEXT:  function_name_id: 2
// CHECK-NEXT:  line: 3
// CHECK-NEXT:  column: 4
// CHECK-NEXT: }
// CHECK-NEXT: stack_frames {
// CHECK-NEXT:  file_location_id: 1
// CHECK-NEXT: }
// CHECK-NEXT: stack_frames {
// CHECK-NEXT:  file_location_id: 2
// CHECK-NEXT:  parent_frame_id: 1
// CHECK-NEXT: }
// CHECK-NEXT: }
#loc = loc(unknown)
module @main attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: !mhlo.token loc(unknown)) -> !mhlo.token {
    %0 = mhlo.after_all %arg0 {xla_shape = "token[]"} : !mhlo.token loc(#type_loc)
    return %0 : !mhlo.token loc(#loc)
  } loc(#loc)
} loc(#loc)
#parent_frame_file_loc = loc("file_name":2:8)
#parent_frame_loc = loc("function_name"(#parent_frame_file_loc))
#child_frame_file_loc = loc("file_name_2":3:4)
#child_frame_loc = loc("function_name_2"(#child_frame_file_loc))
#call_site_loc = loc(callsite(#child_frame_loc at #parent_frame_loc))
#name_loc = loc("name(anothername)"(#call_site_loc))
#type_loc = loc("atype:"(#name_loc))
