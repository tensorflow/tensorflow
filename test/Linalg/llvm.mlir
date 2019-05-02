// RUN: mlir-opt %s -linalg-lower-to-llvm-dialect | FileCheck %s

func @buffer_size(%arg0: !linalg.buffer<f32>) {
  %s = linalg.buffer_size %arg0 : !linalg.buffer<f32>
  return
}
// CHECK-LABEL: func @buffer_size(%arg0: !llvm<"{ float*, i64 }">) {
//       CHECK:   %0 = llvm.extractvalue %arg0[1] : !llvm<"{ float*, i64 }">

func @range(%arg0: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %R = linalg.range %c0:%arg0:%c1 : !linalg.range
  return
}
// CHECK-LABEL: func @range(%arg0: !llvm.i64) {
//       CHECK:   %0 = llvm.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:   %1 = llvm.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:   %2 = llvm.undef : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %3 = llvm.insertvalue %0, %2[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %4 = llvm.insertvalue %arg0, %3[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %5 = llvm.insertvalue %1, %4[2] : !llvm<"{ i64, i64, i64 }">

func @view(%arg0: !linalg.buffer<f32>, %arg1: !linalg.range) {
  %0 = linalg.view %arg0[%arg1] : !linalg.view<?xf32>
  return
}
// CHECK-LABEL: func @view(%arg0: !llvm<"{ float*, i64 }">, %arg1: !llvm<"{ i64, i64, i64 }">) {
//       CHECK:   %0 = llvm.undef : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %1 = llvm.extractvalue %arg0[0] : !llvm<"{ float*, i64 }">
//  CHECK-NEXT:   %2 = llvm.insertvalue %1, %0[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %3 = llvm.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:   %4 = llvm.insertvalue %3, %2[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %5 = llvm.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:   %6 = llvm.extractvalue %arg1[2] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %7 = llvm.mul %5, %6 : !llvm.i64
//  CHECK-NEXT:   %8 = llvm.insertvalue %7, %4[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %9 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %10 = llvm.extractvalue %arg1[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %11 = llvm.sub %10, %9 : !llvm.i64
//  CHECK-NEXT:   %12 = llvm.insertvalue %11, %8[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">

func @slice(%arg0: !linalg.buffer<f32>, %arg1: !linalg.range) {
  %0 = linalg.view %arg0[%arg1] : !linalg.view<?xf32>
  %1 = linalg.slice %0[%arg1] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
  return
}
// CHECK-LABEL: func @slice(%arg0: !llvm<"{ float*, i64 }">, %arg1: !llvm<"{ i64, i64, i64 }">) {
//       CHECK:   %13 = llvm.undef : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %14 = llvm.extractvalue %12[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %15 = llvm.insertvalue %14, %13[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %16 = llvm.extractvalue %12[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %17 = llvm.extractvalue %12[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %18 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %19 = llvm.mul %18, %16 : !llvm.i64
//  CHECK-NEXT:   %20 = llvm.add %17, %19 : !llvm.i64
//  CHECK-NEXT:   %21 = llvm.insertvalue %20, %15[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %22 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %23 = llvm.extractvalue %arg1[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %24 = llvm.sub %23, %22 : !llvm.i64
//  CHECK-NEXT:   %25 = llvm.insertvalue %24, %21[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %26 = llvm.extractvalue %arg1[2] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %27 = llvm.mul %16, %26 : !llvm.i64
//  CHECK-NEXT:   %28 = llvm.insertvalue %27, %25[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">

func @linalg_dot(!llvm<"{ float*, i64, [1 x i64], [1 x i64] }">,
                 !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">,
                 !llvm<"{ float*, i64, [0 x i64], [0 x i64] }">)

func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// CHECK-LABEL: func @dot(%arg0: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, %arg1: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, %arg2: !llvm<"{ float*, i64, [0 x i64], [0 x i64] }">) {
//       CHECK:   llvm.call @linalg_dot(%arg0, %arg1, %arg2) : (!llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, !llvm<"{ float*, i64, [0 x i64], [0 x i64] }">) -> ()
