// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-dynamic-padding | FileCheck %s --dump-input=fail

// Test single argument with padding map lifted to associated encapsulated
// function.
//
// Padding map "\10\02\18\01":
//   arg_index: 0
//   shape_index: 2
//   padding_arg_index: 1
// CHECK-LABEL: func @single_arg_single_shape
func @single_arg_single_shape(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>) {n = 2 : i32} {
    "tf_device.cluster_func"(%ri_0, %ri_1) {func = @func0, padding_map = ["\10\02\18\01"]} : (tensor<i1>, tensor<i1>) -> ()
    tf_device.return
  }
  return
}

// CHECK-LABEL: func @func0
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<i1> {xla_hlo.padding_map = {padding_arg_indices = [1 : i32], shape_indices = [2 : i32]}}, %{{[a-z0-9]+}}: tensor<i1>)
func @func0(%arg0: tensor<i1>, %arg1: tensor<i1>) {
  return
}

// Test single argument with multiple padding maps lifted to associated
// encapsulated function.
//
// Padding map "\10\02\18\01":
//   arg_index: 0
//   shape_index: 2
//   padding_arg_index: 1
//
// Padding map "\10\03\18\02":
//   arg_index: 0
//   shape_index: 3
//   padding_arg_index: 2
// CHECK-LABEL: func @single_arg_multiple_shapes
func @single_arg_multiple_shapes(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>, [%arg0, %arg0] as %ri_2: tensor<i1>) {n = 2 : i32} {
    "tf_device.cluster_func"(%ri_0, %ri_1, %ri_2) {func = @func1, padding_map = ["\10\02\18\01", "\10\03\18\02"]} : (tensor<i1>, tensor<i1>, tensor<i1>) -> ()
    tf_device.return
  }
  return
}

// CHECK-LABEL: func @func1
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<i1> {xla_hlo.padding_map = {padding_arg_indices = [1 : i32, 2 : i32], shape_indices = [2 : i32, 3 : i32]}}, %{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1>)
func @func1(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i1>) {
  return
}

// Test multiple arguments with multiple padding maps lifted to associated
// encapsulated function.
//
// Padding map "\10\02\18\01":
//   arg_index: 0
//   shape_index: 2
//   padding_arg_index: 1
//
// Padding map "\10\03\18\02":
//   arg_index: 0
//   shape_index: 3
//   padding_arg_index: 2
//
// Padding map "\08\04\10\01\18\03":
//   arg_index: 4
//   shape_index: 1
//   padding_arg_index: 3
// CHECK-LABEL: func @multiple_args
func @multiple_args(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>, [%arg0, %arg0] as %ri_2: tensor<i1>, [%arg0, %arg0] as %ri_3: tensor<i1>, [%arg0, %arg0] as %ri_4: tensor<i1>) {n = 2 : i32} {
    "tf_device.cluster_func"(%ri_0, %ri_1, %ri_2, %ri_3, %ri_4) {func = @func2, padding_map = ["\10\02\18\01", "\10\03\18\02", "\08\04\10\01\18\03"]} : (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) -> ()
    tf_device.return
  }
  return
}

// CHECK-LABEL: func @func2
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<i1> {xla_hlo.padding_map = {padding_arg_indices = [1 : i32, 2 : i32], shape_indices = [2 : i32, 3 : i32]}}, %{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1> {xla_hlo.padding_map = {padding_arg_indices = [3 : i32], shape_indices = [1 : i32]}})
func @func2(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i1>, %arg3: tensor<i1>, %arg4: tensor<i1>) {
  return
}

// Test remapping of replicated inputs to encapsulated function arguments.
//
// Padding map "\10\02\18\01":
//   arg_index: 0
//   shape_index: 2
//   padding_arg_index: 1
// CHECK-LABEL: func @remap_indices
func @remap_indices(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>) {n = 2 : i32} {
    "tf_device.cluster_func"(%ri_1, %arg0, %ri_0) {func = @func3, padding_map = ["\10\02\18\01"]} : (tensor<i1>, tensor<i1>, tensor<i1>) -> ()
    tf_device.return
  }
  return
}

// CHECK-LABEL: func @func3
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1> {xla_hlo.padding_map = {padding_arg_indices = [0 : i32], shape_indices = [2 : i32]}})
func @func3(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i1>) {
  return
}

// Test no padding maps are added to encapsulated function if there is no
// replication.
//
// Padding map "\10\02\18\01":
//   arg_index: 0
//   shape_index: 2
//   padding_arg_index: 1
// CHECK-LABEL: func @no_replicate
func @no_replicate(%arg0: tensor<i1>) {
  "tf_device.cluster_func"(%arg0, %arg0, %arg0) {func = @func4, padding_map = ["\10\02\18\01"]} : (tensor<i1>, tensor<i1>, tensor<i1>) -> ()
  return
}

// CHECK-LABEL: func @func4
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1>)
func @func4(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i1>) {
  return
}

// Test encapsulated function is not modified when there are no padding maps.
// CHECK-LABEL: func @no_padding_map
func @no_padding_map(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>) {n = 2 : i32} {
    "tf_device.cluster_func"(%ri_1, %arg0, %ri_0) {func = @func5} : (tensor<i1>, tensor<i1>, tensor<i1>) -> ()
    tf_device.return
  }
  return
}

// CHECK-LABEL: func @func5
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1>)
func @func5(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i1>) {
  return
}

// Test encapsulated function is not modified when padding maps is empty.
// CHECK-LABEL: func @empty_padding_map
func @empty_padding_map(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>) {n = 2 : i32} {
    "tf_device.cluster_func"(%ri_1, %arg0, %ri_0) {func = @func6, padding_map = []} : (tensor<i1>, tensor<i1>, tensor<i1>) -> ()
    tf_device.return
  }
  return
}

// CHECK-LABEL: func @func6
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1>)
func @func6(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i1>) {
  return
}

// Test unused padding map is not added to the encapsulated function.
//
// Padding map "\10\02\18\01":
//   arg_index: 0
//   shape_index: 2
//   padding_arg_index: 1
// CHECK-LABEL: func @unused_padding_map
func @unused_padding_map(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>) {n = 2 : i32} {
    "tf_device.cluster_func"(%ri_1) {func = @func7, padding_map = ["\10\02\18\01"]} : (tensor<i1>) -> ()
    tf_device.return
  }
  return
}

// CHECK-LABEL: func @func7
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<i1>)
func @func7(%arg0: tensor<i1>) {
  return
}

// Test arg that requires a padding arg but padding arg is not an arg to the
// encapsulated function.
//
// Padding map "\10\02\18\01":
//   arg_index: 0
//   shape_index: 2
//   padding_arg_index: 1
//
// Padding map "\08\02\10\02\18\03":
//   arg_index: 2
//   shape_index: 2
//   padding_arg_index: 3
func @missing_padding_arg(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>, [%arg0, %arg0] as %ri_2: tensor<i1>, [%arg0, %arg0] as %ri_3: tensor<i1>) {n = 2 : i32} {
    // expected-warning@+1 {{bad 'padding_map' attribute at index 0, unused padding_arg_index 1}}
    "tf_device.cluster_func"(%ri_0, %ri_2, %ri_3) {func = @func8, padding_map = ["\10\02\18\01", "\08\02\10\02\18\03"]} : (tensor<i1>, tensor<i1>, tensor<i1>) -> ()
    tf_device.return
  }
  return
}

// CHECK-LABEL: func @func8
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<i1>, %{{[a-z0-9]+}}: tensor<i1> {xla_hlo.padding_map = {padding_arg_indices = [2 : i32], shape_indices = [2 : i32]}}, %{{[a-z0-9]+}}: tensor<i1>)
func @func8(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i1>) {
  return
}

// -----

// Test bad padding map attribute (not an array).
func @bad_padding_map() {
  tf_device.replicate {n = 2 : i32} {
    // expected-error@+1 {{'tf_device.cluster_func' op requires 'padding_map' array attribute}}
    "tf_device.cluster_func"() {func = @_func, padding_map = 0 : i32} : () -> ()
    tf_device.return
  }
  return
}

func @_func() {
  return
}

// -----

// Test bad padding map attribute (element in array is not a string).
func @bad_padding_map_element() {
  tf_device.replicate {n = 2 : i32} {
    // expected-error@+1 {{'tf_device.cluster_func' op bad 'padding_map' attribute at index 0, not a string}}
    "tf_device.cluster_func"() {func = @_func, padding_map = [0 : i32]} : () -> ()
    tf_device.return
  }
  return
}

func @_func() {
  return
}

// -----

// Test unparsable padding map.
func @bad_padding_map_proto() {
  tf_device.replicate {n = 2 : i32} {
    // expected-error@+1 {{'tf_device.cluster_func' op bad 'padding_map' attribute at index 0, failed to parse 'z' as tensorflow::tpu::PaddingMap}}
    "tf_device.cluster_func"() {func = @_func, padding_map = ["z"]} : () -> ()
    tf_device.return
  }
  return
}

func @_func() {
  return
}

// -----

// Test negative arg index.
//
// Padding map "\08\FF\FF\FF\FF\FF\FF\FF\FF\FF\01\10\02\18\01":
//   arg_index: -1
//   shape_index: 2
//   padding_arg_index: 1
func @negative_arg_index(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>) {n = 2 : i32} {
    // expected-error@+1 {{'tf_device.cluster_func' op bad 'padding_map' attribute at index 0, arg_index must be in [0, 2), got -1}}
    "tf_device.cluster_func"(%ri_0, %ri_1) {func = @_func, padding_map = ["\08\FF\FF\FF\FF\FF\FF\FF\FF\FF\01\10\02\18\01"]} : (tensor<i1>, tensor<i1>) -> ()
    tf_device.return
  }
  return
}

func @_func(%arg0: tensor<i1>, %arg1: tensor<i1>) {
  return
}

// -----

// Test out of bound arg index.
//
// Padding map "\08\02\10\02\18\01":
//   arg_index: 2
//   shape_index: 2
//   padding_arg_index: 1
func @bad_arg_index(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>) {n = 2 : i32} {
    // expected-error@+1 {{'tf_device.cluster_func' op bad 'padding_map' attribute at index 0, arg_index must be in [0, 2), got 2}}
    "tf_device.cluster_func"(%ri_0, %ri_1) {func = @_func, padding_map = ["\08\02\10\02\18\01"]} : (tensor<i1>, tensor<i1>) -> ()
    tf_device.return
  }
  return
}

func @_func(%arg0: tensor<i1>, %arg1: tensor<i1>) {
  return
}

// -----

// Test negative padding arg index.
//
// Padding map "\08\01\10\02\18\FF\FF\FF\FF\FF\FF\FF\FF\FF\01":
//   arg_index: 1
//   shape_index: 2
//   padding_arg_index: -1
func @negative_padding_arg_index(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>) {n = 2 : i32} {
    // expected-error@+1 {{'tf_device.cluster_func' op bad 'padding_map' attribute at index 0, padding_arg_index must be in [0, 2), got -1}}
    "tf_device.cluster_func"(%ri_0, %ri_1) {func = @_func, padding_map = ["\08\01\10\02\18\FF\FF\FF\FF\FF\FF\FF\FF\FF\01"]} : (tensor<i1>, tensor<i1>) -> ()
    tf_device.return
  }
  return
}

func @_func(%arg0: tensor<i1>, %arg1: tensor<i1>) {
  return
}

// -----

// Test out of bound padding arg index.
//
// Padding map "\08\01\10\02\18\02":
//   arg_index: 1
//   shape_index: 2
//   padding_arg_index: 2
func @bad_padding_arg_index(%arg0: tensor<i1>) {
  tf_device.replicate([%arg0, %arg0] as %ri_0: tensor<i1>, [%arg0, %arg0] as %ri_1: tensor<i1>) {n = 2 : i32} {
    // expected-error@+1 {{'tf_device.cluster_func' op bad 'padding_map' attribute at index 0, padding_arg_index must be in [0, 2), got 2}}
    "tf_device.cluster_func"(%ri_0, %ri_1) {func = @_func, padding_map = ["\08\01\10\02\18\02"]} : (tensor<i1>, tensor<i1>) -> ()
    tf_device.return
  }
  return
}

func @_func(%arg0: tensor<i1>, %arg1: tensor<i1>) {
  return
}
