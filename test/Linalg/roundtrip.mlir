// RUN: mlir-opt %s -verify | mlir-opt | FileCheck %s

func @range(%arg0: index, %arg1: index, %arg2: index) {
  %0 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
  return
}
// CHECK-LABEL: func @range(%arg0: index, %arg1: index, %arg2: index) {
//  CHECK-NEXT:  %0 = linalg.range %arg0:%arg1:%arg2 : !linalg.range

func @buffer(%arg0: i64, %arg1: i64) {
  %0 = muli %arg0, %arg0 : i64
  %1 = linalg.buffer_alloc %0 : !linalg.buffer<f32>
  linalg.buffer_dealloc %1 : !linalg.buffer<f32>
  return
}
// CHECK-LABEL: func @buffer(%arg0: i64, %arg1: i64) {
//  CHECK-NEXT:  %0 = muli %arg0, %arg0 : i64
//  CHECK-NEXT:  %1 = linalg.buffer_alloc %0 : !linalg.buffer<f32>
//  CHECK-NEXT:  linalg.buffer_dealloc %1 : !linalg.buffer<f32>