// RUN: mlir-opt -convert-std-to-spirv %s -o - | FileCheck %s

// CHECK-LABEL: @fmul_scalar
func @fmul_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : f32
  return %0 : f32
}

// CHECK-LABEL: @fmul_vector2
func @fmul_vector2(%arg: vector<2xf32>) -> vector<2xf32> {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @fmul_vector3
func @fmul_vector3(%arg: vector<3xf32>) -> vector<3xf32> {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<3xf32>
  return %0 : vector<3xf32>
}

// CHECK-LABEL: @fmul_vector4
func @fmul_vector4(%arg: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @fmul_vector5
func @fmul_vector5(%arg: vector<5xf32>) -> vector<5xf32> {
  // Vector length of only 2, 3, and 4 is valid for SPIR-V
  // CHECK: mulf
  %0 = mulf %arg, %arg : vector<5xf32>
  return %0 : vector<5xf32>
}

// CHECK-LABEL: @fmul_tensor
func @fmul_tensor(%arg: tensor<4xf32>) -> tensor<4xf32> {
  // For tensors mulf cannot be lowered directly to spv.FMul
  // CHECK: mulf
  %0 = mulf %arg, %arg : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @constval
func @constval() {
  // CHECK: spv.constant true
  %0 = constant true
  // CHECK: spv.constant 42 : i64
  %1 = constant 42
  // CHECK: spv.constant {{[0-9]*\.[0-9]*e?-?[0-9]*}} : f32
  %2 = constant 0.5 : f32
  // CHECK: spv.constant dense<[2, 3]> : vector<2xi32>
  %3 = constant dense<[2, 3]> : vector<2xi32>
  // CHECK: spv.constant 1 : i32
  %4 = constant 1 : index
  return
}

// CHECK-LABEL: @cmpiop
func @cmpiop(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.IEqual
  %0 = cmpi "eq", %arg0, %arg1 : i32
  // CHECK: spv.INotEqual
  %1 = cmpi "ne", %arg0, %arg1 : i32
  // CHECK: spv.SLessThan
  %2 = cmpi "slt", %arg0, %arg1 : i32
  // CHECK: spv.SLessThanEqual
  %3 = cmpi "sle", %arg0, %arg1 : i32
  // CHECK: spv.SGreaterThan
  %4 = cmpi "sgt", %arg0, %arg1 : i32
  // CHECK: spv.SGreaterThanEqual
  %5 = cmpi "sge", %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @select
func @selectOp(%arg0 : i32, %arg1 : i32) {
  %0 = cmpi "sle", %arg0, %arg1 : i32
  // CHECK: spv.Select
  %1 = select %0, %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @div_rem
func @div_rem(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.SDiv
  %0 = divis %arg0, %arg1 : i32
  // CHECK: spv.SMod
  %1 = remis %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @add_sub
func @add_sub(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.IAdd
  %0 = addi %arg0, %arg1 : i32
  // CHECK: spv.ISub
  %1 = subi %arg0, %arg1 : i32
  return
}
