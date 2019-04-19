// RUN: mlir-opt %s -verify | FileCheck %s

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

func @views(%arg0: i64, %arg1: i64, %arg2: index, %arg3: index, %arg4: index) {
  %0 = muli %arg0, %arg0 : i64
  %1 = linalg.buffer_alloc %0 : !linalg.buffer<f32>
  %2 = linalg.range %arg2:%arg3:%arg4 : !linalg.range
  %3 = linalg.base_view %1[%2, %2] : !linalg.view<?x?xf32>
  linalg.buffer_dealloc %1 : !linalg.buffer<f32>
  return
}
// CHECK-LABEL: func @views(%arg0: i64, %arg1: i64, %arg2: index, %arg3: index, %arg4: index) {
//  CHECK-NEXT:  %0 = muli %arg0, %arg0 : i64
//  CHECK-NEXT:  %1 = linalg.buffer_alloc %0 : !linalg.buffer<f32>
//  CHECK-NEXT:  %2 = linalg.range %arg2:%arg3:%arg4 : !linalg.range
//  CHECK-NEXT:  %3 = linalg.base_view %1[%2, %2] : !linalg.view<?x?xf32>
//  CHECK-NEXT:  linalg.buffer_dealloc %1 : !linalg.buffer<f32>