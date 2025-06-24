// RUN: emitters_opt --allow-unregistered-dialect %s -split-input-file -xla-gpu-convert-index-type | FileCheck %s

func.func @addi_default(%arg0: index, %arg1: index) -> index {
  %i0 = arith.addi %arg0, %arg1 : index
  return %i0 : index
}

// CHECK-LABEL: @addi_default
// CHECK-SAME: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK: %[[V1:.*]] = arith.index_castui %[[ARG0]] : index to i64
// CHECK: %[[V2:.*]] = arith.index_castui %[[ARG1]] : index to i64
// CHECK: %[[RI:.*]] = arith.addi %[[V1]], %[[V2]] : i64
// CHECK: %[[R:.*]] = arith.index_castui %[[RI]] : i64 to index
// CHECK: return %[[R]] : index

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @addi_32(%arg0: index, %arg1: index) -> index {
    %i0 = arith.addi %arg0, %arg1 : index
    return %i0 : index
  }
}

// CHECK-LABEL: @addi_32
// CHECK-SAME: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK: %[[V1:.*]] = arith.index_castui %[[ARG0]] : index to i32
// CHECK: %[[V2:.*]] = arith.index_castui %[[ARG1]] : index to i32
// CHECK: %[[RI:.*]] = arith.addi %[[V1]], %[[V2]] : i32
// CHECK: %[[R:.*]] = arith.index_castui %[[RI]] : i32 to index
// CHECK: return %[[R]] : index

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @addi_const(%arg0: index) -> index {
    %c0 = arith.constant 4 : index
    %i0 = arith.addi %c0, %arg0 : index
    return %i0 : index
  }
}

// CHECK-LABEL: @addi_const
// CHECK-SAME: (%[[ARG0:.*]]: index) -> index {
// CHECK: %[[C:.*]] = arith.constant 4 : i32
// CHECK: %[[V1:.*]] = arith.index_castui %[[ARG0]] : index to i32
// CHECK: %[[RI:.*]] = arith.addi %[[V1]], %[[C]] : i32
// CHECK: %[[R:.*]] = arith.index_castui %[[RI]] : i32 to index
// CHECK: return %[[R]] : index

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 8 : i8>>} {
func.func @divui(%arg0: index, %arg1: index) -> index {
  %i0 = arith.divui %arg0, %arg1 : index
  return %i0 : index
}
}

// CHECK-LABEL: @divui
// CHECK-SAME: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK: %[[V1:.*]] = arith.index_castui %[[ARG0]] : index to i8
// CHECK: %[[V2:.*]] = arith.index_castui %[[ARG1]] : index to i8
// CHECK: %[[RI:.*]] = arith.divui %[[V1]], %[[V2]] : i8
// CHECK: %[[R:.*]] = arith.index_castui %[[RI]] : i8 to index
// CHECK: return %[[R]] : index

// -----

func.func @muli(%arg0: index, %arg1: index) -> index {
  %i0 = arith.muli %arg0, %arg1 : index
  return %i0 : index
}

// CHECK-LABEL: @muli
// CHECK-SAME: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK: %[[V1:.*]] = arith.index_castui %[[ARG0]] : index to i64
// CHECK: %[[V2:.*]] = arith.index_castui %[[ARG1]] : index to i64
// CHECK: %[[RI:.*]] = arith.muli %[[V1]], %[[V2]] : i64
// CHECK: %[[R:.*]] = arith.index_castui %[[RI]] : i64 to index
// CHECK: return %[[R]] : index


// -----

func.func @remui(%arg0: index, %arg1: index) -> index {
  %i0 = arith.remui %arg0, %arg1 : index
  return %i0 : index
}

// CHECK-LABEL: @remui
// CHECK-SAME: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK: %[[V1:.*]] = arith.index_castui %[[ARG0]] : index to i64
// CHECK: %[[V2:.*]] = arith.index_castui %[[ARG1]] : index to i64
// CHECK: %[[RI:.*]] = arith.remui %[[V1]], %[[V2]] : i64
// CHECK: %[[R:.*]] = arith.index_castui %[[RI]] : i64 to index
// CHECK: return %[[R]] : index

// -----

func.func @subi(%arg0: index, %arg1: index) -> index {
  %i0 = arith.subi %arg0, %arg1 : index
  return %i0 : index
}

// CHECK-LABEL: @subi
// CHECK-SAME: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK: %[[V1:.*]] = arith.index_castui %[[ARG0]] : index to i64
// CHECK: %[[V2:.*]] = arith.index_castui %[[ARG1]] : index to i64
// CHECK: %[[RI:.*]] = arith.subi %[[V1]], %[[V2]] : i64
// CHECK: %[[R:.*]] = arith.index_castui %[[RI]] : i64 to index
// CHECK: return %[[R]] : index

// -----

func.func @complex(%arg0: index, %arg1: index, %arg2: index) -> index {
  %i0 = arith.subi %arg0, %arg1 : index
  %i1 = arith.addi %arg0, %arg1 : index
  %i2 = arith.muli %i0, %arg2 : index
  %i3 = arith.muli %i1, %arg2 : index
  %i5 = arith.remui %i2, %i3 : index
  return %i5 : index
}

// CHECK-LABEL: @complex
// CHECK: arith.subi %{{.*}} : i64
// CHECK: arith.addi %{{.*}} : i64
// CHECK: arith.muli %{{.*}} : i64
// CHECK: arith.muli %{{.*}} : i64
// CHECK: arith.remui %{{.*}} : i64
// CHECK: %[[R:.*]] = arith.index_castui %{{.*}} : i64 to index
// CHECK: return %[[R]] : index
