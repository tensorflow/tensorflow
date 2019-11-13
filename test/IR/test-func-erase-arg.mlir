// RUN: mlir-opt %s -test-func-erase-arg -split-input-file | FileCheck %s

// CHECK: func @f()
// CHECK-NOT: attributes{{.*}}arg
func @f(%arg0: f32 {test.erase_this_arg}) {
  return
}

// -----

// CHECK: func @f(%arg0: f32 {test.A})
// CHECK-NOT: attributes{{.*}}arg
func @f(
  %arg0: f32 {test.erase_this_arg},
  %arg1: f32 {test.A}) {
  return
}

// -----

// CHECK: func @f(%arg0: f32 {test.A})
// CHECK-NOT: attributes{{.*}}arg
func @f(
  %arg0: f32 {test.A},
  %arg1: f32 {test.erase_this_arg}) {
  return
}

// -----

// CHECK: func @f(%arg0: f32 {test.A}, %arg1: f32 {test.B})
// CHECK-NOT: attributes{{.*}}arg
func @f(
  %arg0: f32 {test.A},
  %arg1: f32 {test.erase_this_arg},
  %arg2: f32 {test.B}) {
  return
}

// -----

// CHECK: func @f(%arg0: f32 {test.A}, %arg1: f32 {test.B})
// CHECK-NOT: attributes{{.*}}arg
func @f(
  %arg0: f32 {test.A},
  %arg1: f32 {test.erase_this_arg},
  %arg2: f32 {test.erase_this_arg},
  %arg3: f32 {test.B}) {
  return
}

// -----

// CHECK: func @f(%arg0: f32 {test.A}, %arg1: f32 {test.B}, %arg2: f32 {test.C})
// CHECK-NOT: attributes{{.*}}arg
func @f(
  %arg0: f32 {test.A},
  %arg1: f32 {test.erase_this_arg},
  %arg2: f32 {test.B},
  %arg3: f32 {test.erase_this_arg},
  %arg4: f32 {test.C}) {
  return
}

// -----

// CHECK: func @f(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>, %arg2: tensor<3xf32>)
// CHECK-NOT: attributes{{.*}}arg
func @f(
  %arg0: tensor<1xf32>,
  %arg1: f32 {test.erase_this_arg},
  %arg2: tensor<2xf32>,
  %arg3: f32 {test.erase_this_arg},
  %arg4: tensor<3xf32>) {
  return
}
