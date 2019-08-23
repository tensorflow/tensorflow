// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.IEqual
//===----------------------------------------------------------------------===//

func @iequal_scalar(%arg0: i32, %arg1: i32) -> i1 {
  // CHECK: spv.IEqual {{.*}}, {{.*}} : i32
  %0 = spv.IEqual %arg0, %arg1 : i32
  return %0 : i1
}

// -----

func @iequal_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.IEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.IEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.INotEqual
//===----------------------------------------------------------------------===//

func @inotequal_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.INotEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.INotEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.IMul
//===----------------------------------------------------------------------===//

func @imul_scalar(%arg: i32) -> i32 {
  // CHECK: spv.IMul
  %0 = spv.IMul %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.SGreaterThan
//===----------------------------------------------------------------------===//

func @sgt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.SGreaterThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SGreaterThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.SGreaterThanEqual
//===----------------------------------------------------------------------===//

func @sge_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.SGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SGreaterThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.SLessThan
//===----------------------------------------------------------------------===//

func @slt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.SLessThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SLessThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.SLessThanEqual
//===----------------------------------------------------------------------===//

func @slte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.SLessThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SLessThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.UGreaterThan
//===----------------------------------------------------------------------===//

func @ugt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.UGreaterThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.UGreaterThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.UGreaterThanEqual
//===----------------------------------------------------------------------===//

func @ugte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.UGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.UGreaterThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.ULessThan
//===----------------------------------------------------------------------===//

func @ult_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.ULessThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.ULessThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.ULessThanEqual
//===----------------------------------------------------------------------===//

func @ulte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.ULessThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.ULessThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}
