// RUN: tf-mlir-translate -split-input-file -mlir-hlo-to-hlo %s | FileCheck %s
// RUN: tf-mlir-translate -split-input-file -mlir-hlo-to-hlo -emit-use-tuple-args %s | FileCheck %s --check-prefix=TUPLE

// Test entry function with no dynamic parameter bindings.

func @main(%arg0: tensor<10xf32>, %arg1: tensor<i32>) {
  return
}

// CHECK-LABEL: hlo_module
// CHECK:      dynamic_parameter_binding
// CHECK-NEXT: }

// TUPLE-LABEL: hlo_module
// TUPLE:      dynamic_parameter_binding
// TUPLE-NEXT: }

// -----

// Test entry function with single dynamic parameter binding on an argument.

func @main(%arg0: tensor<10xf32> {xla_hlo.padding_map = {shape_indices = [0 : i32], padding_arg_indices = [1 : i32]}}, %arg1: tensor<i32>) {
  return
}

// CHECK-LABEL: hlo_module
// CHECK:      dynamic_parameter_binding
// CHECK-NEXT:   entries
// CHECK-NEXT:     dynamic_param_num: 1
// CHECK-NEXT:   }
// CHECK-NOT:    entries

// TUPLE-LABEL: hlo_module
// TUPLE:      dynamic_parameter_binding
// TUPLE-NEXT:   entries
// TUPLE-NEXT:     dynamic_param_index: 1
// TUPLE-NEXT:     target_param_index: 0
// TUPLE-NEXT:   }
// TUPLE-NOT:    entries

// -----

// Test entry function with multiple dynamic parameter bindings on an argument.

func @main(%arg0: tensor<8x10xf32> {xla_hlo.padding_map = {shape_indices = [0 : i32, 1 : i32], padding_arg_indices = [1 : i32, 2 : i32]}}, %arg1: tensor<i32>, %arg2: tensor<i32>) {
  return
}

// CHECK-LABEL: hlo_module
// CHECK:      dynamic_parameter_binding
// CHECK-NEXT:   entries
// CHECK-NEXT:     dynamic_param_num: 1
// CHECK-NEXT:   }
// CHECK-NEXT:   entries
// CHECK-NEXT:     dynamic_param_num: 2
// CHECK-NEXT:     target_param_dim_num: 1
// CHECK-NEXT:   }
// CHECK-NOT:    entries

// TUPLE-LABEL: hlo_module
// TUPLE:      dynamic_parameter_binding
// TUPLE-NEXT:   entries
// TUPLE-NEXT:     dynamic_param_index: 1
// TUPLE-NEXT:     target_param_index: 0
// TUPLE-NEXT:   }
// TUPLE-NEXT:   entries
// TUPLE-NEXT:     dynamic_param_index: 2
// TUPLE-NEXT:     target_param_index: 0
// TUPLE-NEXT:     target_param_dim_num: 1
// TUPLE-NEXT:   }
// TUPLE-NOT:    entries

// -----

// Test entry function with multiple dynamic parameter bindings on multiple
// arguments.

func @main(%arg0: tensor<8x10xf32> {xla_hlo.padding_map = {shape_indices = [0 : i32, 1 : i32], padding_arg_indices = [1 : i32, 2 : i32]}}, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<10x8x6xi32> {xla_hlo.padding_map = {shape_indices = [2 : i32], padding_arg_indices = [4 : i32]}}, %arg4: tensor<i32>) {
  return
}

// CHECK-LABEL: hlo_module
// CHECK:      dynamic_parameter_binding
// CHECK-NEXT:   entries
// CHECK-NEXT:     dynamic_param_num: 1
// CHECK-NEXT:   }
// CHECK-NEXT:   entries
// CHECK-NEXT:     dynamic_param_num: 2
// CHECK-NEXT:     target_param_dim_num: 1
// CHECK-NEXT:   }
// CHECK-NEXT:   entries
// CHECK-NEXT:     dynamic_param_num: 4
// CHECK-NEXT:     target_param_num: 3
// CHECK-NEXT:     target_param_dim_num: 2
// CHECK-NEXT:   }
// CHECK-NOT:    entries

// TUPLE-LABEL: hlo_module
// TUPLE:      dynamic_parameter_binding
// TUPLE-NEXT:   entries
// TUPLE-NEXT:     dynamic_param_index: 1
// TUPLE-NEXT:     target_param_index: 0
// TUPLE-NEXT:   }
// TUPLE-NEXT:   entries
// TUPLE-NEXT:     dynamic_param_index: 2
// TUPLE-NEXT:     target_param_index: 0
// TUPLE-NEXT:     target_param_dim_num: 1
// TUPLE-NEXT:   }
// TUPLE-NEXT:   entries
// TUPLE-NEXT:     dynamic_param_index: 4
// TUPLE-NEXT:     target_param_index: 3
// TUPLE-NEXT:     target_param_dim_num: 2
// TUPLE-NEXT:   }
// TUPLE-NOT:    entries
