// RUN: not tf-mlir-translate -split-input-file -mlir-hlo-to-hlo %s -o - 2>&1 | FileCheck %s

// Test bad `mhlo.padding_map` attribute type.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = ""}) {
  return
}

// CHECK: requires 'mhlo.padding_map' dict attribute at arg 1

// -----

// Test missing `shape_indices` attribute in `mhlo.padding_map`.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {}}) {
  return
}

// CHECK: requires 'shape_indices' array attribute in 'mhlo.padding_map' dict at arg 1

// -----

// Test bad `shape_indices` attribute type in `mhlo.padding_map`.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = ""}}) {
  return
}

// CHECK: requires 'shape_indices' array attribute in 'mhlo.padding_map' dict at arg 1

// -----

// Test missing `padding_arg_indices` attribute in `mhlo.padding_map`.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = []}}) {
  return
}

// CHECK: requires 'padding_arg_indices' array attribute in 'mhlo.padding_map' dict at arg 1

// -----

// Test bad `padding_arg_indices` attribute type in `mhlo.padding_map`.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = [], padding_arg_indices = ""}}) {
  return
}

// CHECK: requires 'padding_arg_indices' array attribute in 'mhlo.padding_map' dict at arg 1

// -----

// Test mismatched `shape_indices` and `padding_arg_indices` lengths.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = [ 0: i32 ], padding_arg_indices = [ 0: i32, 0 : i32 ]}}) {
  return
}

// CHECK: requires 'shape_indices' and 'padding_arg_indices' array attributes in 'mhlo.padding_map' dic at arg 1 to be of the same size, got sizes 1 and 2

// -----

// Test non integer attribute in `shape_indices`.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = [ 0: i32, 0.0: f32 ], padding_arg_indices = [ 0: i32, 0: i32 ]}}) {
  return
}

// CHECK: requires element 1 in 'shape_indices' array of 'mhlo.padding_map' dict at arg 1 to be an int attribute

// -----

// Test non integer attribute in `padding_arg_indices`.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = [ 0: i32, 0: i32 ], padding_arg_indices = [ 0: i32, 0.0: f32 ]}}) {
  return
}

// CHECK: requires element 1 in 'padding_arg_indices' array of 'mhlo.padding_map' dict at arg 1 to be an int attribute

// -----

// Test negative out of range shape index in `shape_indices`.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = [ -1: i32 ], padding_arg_indices = [ 0: i32 ]}}) {
  return
}

// CHECK: requires element 0 in 'shape_indices' array of 'mhlo.padding_map' dict at arg 1 to be in range [0, 1), got -1

// -----

// Test positive out of range shape index in `shape_indices`.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = [ 1: i32 ], padding_arg_indices = [ 0: i32 ]}}) {
  return
}

// CHECK: requires element 0 in 'shape_indices' array of 'mhlo.padding_map' dict at arg 1 to be in range [0, 1), got 1

// -----

// Test negative shape index in `shape_indices` for unranked argument.

func @main(%arg0: tensor<i32>, %arg1: tensor<*xf32> {mhlo.padding_map = {shape_indices = [ -1: i32 ], padding_arg_indices = [ 0: i32 ]}}) {
  return
}

// CHECK: requires element 0 in 'shape_indices' array of 'mhlo.padding_map' dict at arg 1 to be non-negative, got -1

// -----

// Test duplicate shape indices in `shape_indices`.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = [ 0: i32, 0: i32 ], padding_arg_indices = [ 0: i32, 0: i32 ]}}) {
  return
}

// CHECK: requires elements in 'shape_indices' array of 'mhlo.padding_map' dict at arg 1 to be unique, got duplicate element 0 at index 1

// -----

// Test negative out of range shape index in `padding_arg_indices`.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = [ 0: i32 ], padding_arg_indices = [ -1: i32 ]}}) {
  return
}

// CHECK: requires element 0 in 'padding_arg_indices' array of 'mhlo.padding_map' dict at arg 1 to be in range [0, 2), got -1

// -----

// Test positive out of range shape index in `padding_arg_indices`.

func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = [ 0: i32 ], padding_arg_indices = [ 2: i32 ]}}) {
  return
}

// CHECK: requires element 0 in 'padding_arg_indices' array of 'mhlo.padding_map' dict at arg 1 to be in range [0, 2), got 2

// -----

// Test non scalar padding argument.

func @main(%arg0: tensor<8xi32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = [ 0: i32 ], padding_arg_indices = [ 0: i32 ]}}) {
  return
}

// CHECK: requires arg 0 to be a scalar for use as a dynamic parameter

// -----

// Test non integer type padding argument.

func @main(%arg0: tensor<f32>, %arg1: tensor<10xf32> {mhlo.padding_map = {shape_indices = [ 0: i32 ], padding_arg_indices = [ 0: i32 ]}}) {
  return
}

// CHECK: requires arg 0 to be of an int type for use as a dynamic parameter
