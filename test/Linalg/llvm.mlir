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

func @view3d(%arg0: !linalg.buffer<f32>, %arg1: !linalg.range, %arg2: !linalg.range, %arg3: !linalg.range) {
  %0 = linalg.view %arg0[%arg1, %arg2, %arg3] : !linalg.view<?x?x?xf32>
  return
}
// CHECK-LABEL: func @view3d(%arg0: !llvm<"{ float*, i64 }">, %arg1: !llvm<"{ i64, i64, i64 }">, %arg2: !llvm<"{ i64, i64, i64 }">, %arg3: !llvm<"{ i64, i64, i64 }">) {
//       CHECK:   %5 = llvm.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:   %6 = llvm.extractvalue %arg3[2] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %7 = llvm.mul %5, %6 : !llvm.i64
//  CHECK-NEXT:   %8 = llvm.insertvalue %7, %4[3, 2] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   %10 = llvm.extractvalue %arg3[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %13 = llvm.mul %5, %10 : !llvm.i64
//  CHECK-NEXT:   %14 = llvm.extractvalue %arg2[2] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %15 = llvm.mul %13, %14 : !llvm.i64
//  CHECK-NEXT:   %16 = llvm.insertvalue %15, %12[3, 1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">

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

func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// CHECK-LABEL: func @dot(%arg0: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, %arg1: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, %arg2: !llvm<"{ float*, i64, [0 x i64], [0 x i64] }">) {
//       CHECK:   llvm.call @linalg_dot(%arg0, %arg1, %arg2) : (!llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, !llvm<"{ float*, i64, [0 x i64], [0 x i64] }">) -> ()

func @dim(%arg0: !linalg.view<?x?xf32>) {
  %0 = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
  return
}
// CHECK-LABEL: func @dim(%arg0: !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">) {
//       CHECK:   %0 = llvm.extractvalue %arg0[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">

func @range_intersect(%arg0: !linalg.range, %arg1: !linalg.range) -> !linalg.range {
  %0 = linalg.range_intersect %arg0, %arg1 : !linalg.range
  return %0 : !linalg.range
}
// CHECK-LABEL: func @range_intersect(%arg0: !llvm<"{ i64, i64, i64 }">, %arg1: !llvm<"{ i64, i64, i64 }">) -> !llvm<"{ i64, i64, i64 }"> {
//       CHECK:   %0 = llvm.extractvalue %arg0[0] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %1 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %2 = llvm.extractvalue %arg0[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %3 = llvm.extractvalue %arg1[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %4 = llvm.extractvalue %arg0[2] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %5 = llvm.extractvalue %arg1[2] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %6 = llvm.undef : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %7 = llvm.icmp "sge" %0, %1 : !llvm.i64
//       CHECK:   %8 = llvm.select %7, %0, %1 : !llvm.i1, !llvm.i64
//       CHECK:   %9 = llvm.insertvalue %8, %6[0] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %10 = llvm.icmp "sle" %2, %3 : !llvm.i64
//       CHECK:   %11 = llvm.select %10, %2, %3 : !llvm.i1, !llvm.i64
//       CHECK:   %12 = llvm.insertvalue %11, %9[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %13 = llvm.mul %4, %5 : !llvm.i64
//       CHECK:   %14 = llvm.insertvalue %13, %12[2] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   llvm.return %14 : !llvm<"{ i64, i64, i64 }">

func @linalg_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  linalg.for %i0 = %arg0 to %arg1 step %arg2 {
    %a = muli %i0, %arg0 : index
  }
  return
}
// CHECK-LABEL: func @linalg_for(%arg0: !llvm.i64, %arg1: !llvm.i64, %arg2: !llvm.i64) {
//       CHECK:   llvm.br ^bb2(%arg0 : !llvm.i64)
//       CHECK: ^bb1:   // pred: ^bb2
//       CHECK:   llvm.return
//       CHECK: ^bb2(%0: !llvm.i64):    // 2 preds: ^bb0, ^bb3
//       CHECK:   %1 = llvm.icmp "sgt" %arg1, %0 : !llvm.i64
//       CHECK:   llvm.cond_br %1, ^bb3, ^bb1
//       CHECK: ^bb3:   // pred: ^bb2
//       CHECK:   %2 = llvm.mul %0, %arg0 : !llvm.i64
//       CHECK:   %3 = llvm.add %0, %arg2 : !llvm.i64
//       CHECK:   llvm.br ^bb2(%3 : !llvm.i64)

func @linalg_for_2(%arg0 : index, %arg1 : index, %arg2 : index) {
  linalg.for %i0 = %arg0 to %arg1 step %arg2 {
    linalg.for %i1 = %arg0 to %arg1 step %arg2 {
      %a = muli %i0, %i1 : index
    }
    linalg.for %i2 = %arg0 to %arg1 step %arg2 {
      %b = muli %i0, %i2 : index
    }
  }
  return
}
// CHECK-LABEL: func @linalg_for_2(%arg0: !llvm.i64, %arg1: !llvm.i64, %arg2: !llvm.i64) {
//       CHECK:   llvm.br ^bb2(%arg0 : !llvm.i64)
//       CHECK: ^bb1:   // pred: ^bb2
//       CHECK:   llvm.return
//       CHECK: ^bb2(%0: !llvm.i64):    // 2 preds: ^bb0, ^bb5
//       CHECK:   %1 = llvm.icmp "sgt" %arg1, %0 : !llvm.i64
//       CHECK:   llvm.cond_br %1, ^bb3, ^bb1
//       CHECK: ^bb3:   // pred: ^bb2
//       CHECK:   llvm.br ^bb8(%arg0 : !llvm.i64)
//       CHECK: ^bb4:   // pred: ^bb8
//       CHECK:   llvm.br ^bb6(%arg0 : !llvm.i64)
//       CHECK: ^bb5:   // pred: ^bb6
//       CHECK:   %2 = llvm.add %0, %arg2 : !llvm.i64
//       CHECK:   llvm.br ^bb2(%2 : !llvm.i64)
//       CHECK: ^bb6(%3: !llvm.i64):    // 2 preds: ^bb4, ^bb7
//       CHECK:   %4 = llvm.icmp "sgt" %arg1, %3 : !llvm.i64
//       CHECK:   llvm.cond_br %4, ^bb7, ^bb5
//       CHECK: ^bb7:   // pred: ^bb6
//       CHECK:   %5 = llvm.mul %0, %3 : !llvm.i64
//       CHECK:   %6 = llvm.add %3, %arg2 : !llvm.i64
//       CHECK:   llvm.br ^bb6(%6 : !llvm.i64)
//       CHECK: ^bb8(%7: !llvm.i64):    // 2 preds: ^bb3, ^bb9
//       CHECK:   %8 = llvm.icmp "sgt" %arg1, %7 : !llvm.i64
//       CHECK:   llvm.cond_br %8, ^bb9, ^bb4
//       CHECK: ^bb9:   // pred: ^bb8
//       CHECK:   %9 = llvm.mul %0, %7 : !llvm.i64
//       CHECK:   %10 = llvm.add %7, %arg2 : !llvm.i64
//       CHECK:   llvm.br ^bb8(%10 : !llvm.i64)

func @subview(%arg0: !linalg.view<?x?xf32>) {
  %c0 = constant 0 : index
  %0 = linalg.subview %arg0[%c0, %c0, %c0, %c0, %c0, %c0] : !linalg.view<?x?xf32>
  return
}
// CHECK-LABEL: func @subview(%arg0: !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">) {
//       CHECK:   %0 = llvm.constant(0 : index) : !llvm.i64
//       CHECK:   %1 = llvm.extractvalue %arg0[2, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   %2 = llvm.icmp "slt" %1, %0 : !llvm.i64
//       CHECK:   %3 = llvm.select %2, %1, %0 : !llvm.i1, !llvm.i64
//       CHECK:   %4 = llvm.undef : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %5 = llvm.insertvalue %0, %4[0] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %6 = llvm.insertvalue %3, %5[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %7 = llvm.insertvalue %0, %6[2] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %8 = llvm.extractvalue %arg0[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   %9 = llvm.icmp "slt" %8, %0 : !llvm.i64
//       CHECK:   %10 = llvm.select %9, %8, %0 : !llvm.i1, !llvm.i64
//       CHECK:   %11 = llvm.undef : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %12 = llvm.insertvalue %0, %11[0] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %13 = llvm.insertvalue %10, %12[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %14 = llvm.insertvalue %0, %13[2] : !llvm<"{ i64, i64, i64 }">
