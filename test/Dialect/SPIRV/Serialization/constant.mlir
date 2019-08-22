// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK-LABEL: @bool_const
  func @bool_const() -> () {
    // CHECK: spv.constant true
    %0 = spv.constant true
    // CHECK: spv.constant false
    %1 = spv.constant false

    %2 = spv.Variable init(%0): !spv.ptr<i1, Function>
    %3 = spv.Variable init(%1): !spv.ptr<i1, Function>
    spv.Return
  }

  // CHECK-LABEL: @i32_const
  func @i32_const() -> () {
    // CHECK: spv.constant 0 : i32
    %0 = spv.constant  0 : i32
    // CHECK: spv.constant 10 : i32
    %1 = spv.constant 10 : i32
    // CHECK: spv.constant -5 : i32
    %2 = spv.constant -5 : i32

    %3 = spv.IAdd %0, %1 : i32
    %4 = spv.IAdd %2, %3 : i32
    spv.Return
  }

  // CHECK-LABEL: @i64_const
  func @i64_const() -> () {
    // CHECK: spv.constant 4294967296 : i64
    %0 = spv.constant           4294967296 : i64 //  2^32
    // CHECK: spv.constant -4294967296 : i64
    %1 = spv.constant          -4294967296 : i64 // -2^32
    // CHECK: spv.constant 9223372036854775807 : i64
    %2 = spv.constant  9223372036854775807 : i64 //  2^63 - 1
    // CHECK: spv.constant -9223372036854775808 : i64
    %3 = spv.constant -9223372036854775808 : i64 // -2^63

    %4 = spv.IAdd %0, %1 : i64
    %5 = spv.IAdd %2, %3 : i64
    spv.Return
  }

  // CHECK-LABEL: @i16_const
  func @i16_const() -> () {
    // CHECK: spv.constant -32768 : i16
    %0 = spv.constant -32768 : i16 // -2^15
    // CHECK: spv.constant 32767 : i16
    %1 = spv.constant 32767 : i16 //  2^15 - 1

    %2 = spv.IAdd %0, %1 : i16
    spv.Return
  }

  // CHECK-LABEL: @float_const
  func @float_const() -> () {
    // CHECK: spv.constant 0.000000e+00 : f32
    %0 = spv.constant 0. : f32
    // CHECK: spv.constant 1.000000e+00 : f32
    %1 = spv.constant 1. : f32
    // CHECK: spv.constant -0.000000e+00 : f32
    %2 = spv.constant -0. : f32
    // CHECK: spv.constant -1.000000e+00 : f32
    %3 = spv.constant -1. : f32
    // CHECK: spv.constant 7.500000e-01 : f32
    %4 = spv.constant 0.75 : f32
    // CHECK: spv.constant -2.500000e-01 : f32
    %5 = spv.constant -0.25 : f32

    %6 = spv.FAdd %0, %1 : f32
    %7 = spv.FAdd %2, %3 : f32
    %8 = spv.FAdd %4, %5 : f32
    spv.Return
  }

  // CHECK-LABEL: @double_const
  func @double_const() -> () {
    // TODO(antiagainst): test range boundary values
    // CHECK: spv.constant 1.024000e+03 : f64
    %0 = spv.constant 1024. : f64
    // CHECK: spv.constant -1.024000e+03 : f64
    %1 = spv.constant -1024. : f64

    %2 = spv.FAdd %0, %1 : f64
    spv.Return
  }

  // CHECK-LABEL: @half_const
  func @half_const() -> () {
    // CHECK: spv.constant 5.120000e+02 : f16
    %0 = spv.constant 512. : f16
    // CHECK: spv.constant -5.120000e+02 : f16
    %1 = spv.constant -512. : f16

    %2 = spv.FAdd %0, %1 : f16
    spv.Return
  }

  // CHECK-LABEL: @bool_vector_const
  func @bool_vector_const() -> () {
    // CHECK: spv.constant dense<false> : vector<2xi1>
    %0 = spv.constant dense<false> : vector<2xi1>
    // CHECK: spv.constant dense<[true, true, true]> : vector<3xi1>
    %1 = spv.constant dense<true> : vector<3xi1>
    // CHECK: spv.constant dense<[false, true]> : vector<2xi1>
    %2 = spv.constant dense<[false, true]> : vector<2xi1>

    %3 = spv.Variable init(%0): !spv.ptr<vector<2xi1>, Function>
    %4 = spv.Variable init(%1): !spv.ptr<vector<3xi1>, Function>
    %5 = spv.Variable init(%2): !spv.ptr<vector<2xi1>, Function>
    spv.Return
  }

  // CHECK-LABEL: @int_vector_const
  func @int_vector_const() -> () {
    // CHECK: spv.constant dense<0> : vector<3xi32>
    %0 = spv.constant dense<0> : vector<3xi32>
    // CHECK: spv.constant dense<1> : vector<3xi32>
    %1 = spv.constant dense<1> : vector<3xi32>
    // CHECK: spv.constant dense<[2, -3, 4]> : vector<3xi32>
    %2 = spv.constant dense<[2, -3, 4]> : vector<3xi32>

    %3 = spv.IAdd %0, %1 : vector<3xi32>
    %4 = spv.IAdd %2, %3 : vector<3xi32>
    spv.Return
  }

  // CHECK-LABEL: @fp_vector_const
  func @fp_vector_const() -> () {
    // CHECK: spv.constant dense<0.000000e+00> : vector<4xf32>
    %0 = spv.constant dense<0.> : vector<4xf32>
    // CHECK: spv.constant dense<-1.500000e+01> : vector<4xf32>
    %1 = spv.constant dense<-15.> : vector<4xf32>
    // CHECK: spv.constant dense<[7.500000e-01, -2.500000e-01, 1.000000e+01, 4.200000e+01]> : vector<4xf32>
    %2 = spv.constant dense<[0.75, -0.25, 10., 42.]> : vector<4xf32>

    %3 = spv.FAdd %0, %1 : vector<4xf32>
    %4 = spv.FAdd %2, %3 : vector<4xf32>
    spv.Return
  }

  // CHECK-LABEL: @array_const
  func @array_const() -> (!spv.array<2 x vector<2xf32>>) {
    // CHECK: spv.constant [dense<3.000000e+00> : vector<2xf32>, dense<[4.000000e+00, 5.000000e+00]> : vector<2xf32>] : !spv.array<2 x vector<2xf32>>
    %0 = spv.constant [dense<3.0> : vector<2xf32>, dense<[4., 5.]> : vector<2xf32>] : !spv.array<2 x vector<2xf32>>

    spv.ReturnValue %0 : !spv.array<2 x vector<2xf32>>
  }

  // CHECK-LABEL: @ignore_not_used_const
  func @ignore_not_used_const() -> () {
    %0 = spv.constant false
    // CHECK-NEXT: spv.Return
    spv.Return
  }

  // CHECK-LABEL: @materialize_const_at_each_use
  func @materialize_const_at_each_use() -> (i32) {
    // CHECK: %[[USE1:.*]] = spv.constant 42 : i32
    // CHECK: %[[USE2:.*]] = spv.constant 42 : i32
    // CHECK: spv.IAdd %[[USE1]], %[[USE2]]
    %0 = spv.constant 42 : i32
    %1 = spv.IAdd %0, %0 : i32
    spv.ReturnValue %1 : i32
  }
}
