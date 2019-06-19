// RUN: linalg1-opt %s | FileCheck %s
// RUN: linalg1-opt %s -lower-linalg-to-llvm | FileCheck %s -check-prefix=LLVM

func @view_op(%arg0: memref<f32>, %arg1: memref<?xf32>, %arg2: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %c17 = constant 17 : index
  %c1 = constant 1 : index
  %3 = linalg.range %c3:%c17:%c1 : !linalg.range
  %4 = linalg.view %arg0[] : memref<f32>, !linalg.view<f32>
  %5 = linalg.view %arg1[%3] : memref<?xf32>, !linalg.range, !linalg.view<?xf32>
  %6 = linalg.view %arg2[%3, %3] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  "some_consumer"(%4, %5, %6) : (!linalg.view<f32>, !linalg.view<?xf32>, !linalg.view<?x?xf32>) -> ()
  return
}
// CHECK-LABEL: func @view_op(%arg0: memref<f32>, %arg1: memref<?xf32>, %arg2: memref<?x?xf32>) {
//       CHECK:  %0 = linalg.range {{.*}} : !linalg.range
//       CHECK:  {{.*}} = linalg.view %arg0[] : memref<f32>, !linalg.view<f32>
//       CHECK:  {{.*}} = linalg.view %arg1[%0] : memref<?xf32>, !linalg.range, !linalg.view<?xf32>
//       CHECK:  {{.*}} = linalg.view %arg2[%0, %0] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>

func @slice_op(%arg0: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %1 = dim %arg0, 0 : memref<?x?xf32>
  %2 = dim %arg0, 1 : memref<?x?xf32>
  %3 = linalg.range %c0:%1:%c1 : !linalg.range
  %4 = linalg.range %c0:%2:%c1 : !linalg.range
  %5 = linalg.view %arg0[%3, %4] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  affine.for %i0 = 0 to (d0) -> (d0)(%1) {
    affine.for %i1 = 0 to (d0) -> (d0)(%2) {
      %6 = linalg.slice %5[%i0] {dim: 1} : !linalg.view<?x?xf32>, index
      "some_consumer"(%6) : (!linalg.view<?xf32>) -> ()
      %7 = linalg.slice %5[%i1] {dim: 0} : !linalg.view<?x?xf32>, index
      %8 = linalg.slice %7[%i0] {dim: 0} : !linalg.view<?xf32>, index
    }
  }
  return
}
// CHECK-LABEL: func @slice_op(%arg0: memref<?x?xf32>) {
//       CHECK:  %[[M:.*]] = dim %arg0, 0 : memref<?x?xf32>
//       CHECK:  %[[N:.*]] = dim %arg0, 1 : memref<?x?xf32>
//       CHECK:  %[[r1:.*]] = linalg.range %c0:%[[M]]:%c1 : !linalg.range
//       CHECK:  %[[r2:.*]] = linalg.range %c0:%[[N]]:%c1 : !linalg.range
//       CHECK:  %[[V:.*]] = linalg.view %arg0[%[[r1]], %[[r2]]] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//       CHECK:  affine.for %i0 = 0 to #map1(%0) {
//       CHECK:    affine.for %i1 = 0 to #map1(%1) {
//       CHECK:      {{.*}} = linalg.slice %[[V]][%i0] {dim: 1} : !linalg.view<?x?xf32>, index
//       CHECK:      %[[V2:.*]] = linalg.slice %[[V]][%i1] {dim: 0} : !linalg.view<?x?xf32>, index
//       CHECK:      {{.*}} = linalg.slice %[[V2]][%i0] {dim: 0} : !linalg.view<?xf32>, index

func @rangeConversion(%arg0: index, %arg1: index, %arg2: index) {
  %0 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
  return
}
// LLVM-LABEL: @rangeConversion
// LLVM-NEXT: %0 = llvm.undef : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %1 = llvm.insertvalue %arg0, %0[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %2 = llvm.insertvalue %arg1, %1[1] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %3 = llvm.insertvalue %arg2, %2[2] : !llvm<"{ i64, i64, i64 }">

func @viewRangeConversion(%arg0: memref<?x?xf32>, %arg1: !linalg.range, %arg2: !linalg.range) {
  %0 = linalg.view %arg0[%arg1, %arg2] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  return
}
// LLVM-LABEL: @viewRangeConversion
// LLVM-NEXT: %0 = llvm.undef : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %1 = llvm.extractvalue %arg0[0] : !llvm<"{ float*, i64, i64 }">
// LLVM-NEXT: %2 = llvm.insertvalue %1, %0[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %3 = llvm.extractvalue %arg0[2] : !llvm<"{ float*, i64, i64 }">
// LLVM-NEXT: %4 = llvm.constant(1 : index) : !llvm.i64
// LLVM-NEXT: %5 = llvm.mul %4, %3 : !llvm.i64
// LLVM-NEXT: %6 = llvm.constant(0 : index) : !llvm.i64
// LLVM-NEXT: %7 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %8 = llvm.mul %7, %5 : !llvm.i64
// LLVM-NEXT: %9 = llvm.add %6, %8 : !llvm.i64
// LLVM-NEXT: %10 = llvm.extractvalue %arg2[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %11 = llvm.mul %10, %4 : !llvm.i64
// LLVM-NEXT: %12 = llvm.add %9, %11 : !llvm.i64
// LLVM-NEXT: %13 = llvm.insertvalue %12, %2[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %14 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %15 = llvm.extractvalue %arg1[1] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %16 = llvm.sub %15, %14 : !llvm.i64
// LLVM-NEXT: %17 = llvm.insertvalue %16, %13[2, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %18 = llvm.extractvalue %arg2[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %19 = llvm.extractvalue %arg2[1] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %20 = llvm.sub %19, %18 : !llvm.i64
// LLVM-NEXT: %21 = llvm.insertvalue %20, %17[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %22 = llvm.extractvalue %arg1[2] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %23 = llvm.mul %5, %22 : !llvm.i64
// LLVM-NEXT: %24 = llvm.insertvalue %23, %21[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %25 = llvm.extractvalue %arg2[2] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %26 = llvm.mul %4, %25 : !llvm.i64
// LLVM-NEXT: %27 = llvm.insertvalue %26, %24[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">

func @viewNonRangeConversion(%arg0: memref<?x?xf32>, %arg1: !linalg.range, %arg2: index) {
  %0 = linalg.view %arg0[%arg1, %arg2] : memref<?x?xf32>, !linalg.range, index, !linalg.view<?xf32>
  return
}
// LLVM-LABEL: @viewNonRangeConversion
// LLVM-NEXT: %0 = llvm.undef : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: %1 = llvm.extractvalue %arg0[0] : !llvm<"{ float*, i64, i64 }">
// LLVM-NEXT: %2 = llvm.insertvalue %1, %0[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: %3 = llvm.extractvalue %arg0[2] : !llvm<"{ float*, i64, i64 }">
// LLVM-NEXT: %4 = llvm.constant(1 : index) : !llvm.i64
// LLVM-NEXT: %5 = llvm.mul %4, %3 : !llvm.i64
// LLVM-NEXT: %6 = llvm.constant(0 : index) : !llvm.i64
// LLVM-NEXT: %7 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %8 = llvm.mul %7, %5 : !llvm.i64
// LLVM-NEXT: %9 = llvm.add %6, %8 : !llvm.i64
// LLVM-NEXT: %10 = llvm.mul %arg2, %4 : !llvm.i64
// LLVM-NEXT: %11 = llvm.add %9, %10 : !llvm.i64
// LLVM-NEXT: %12 = llvm.insertvalue %11, %2[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: %13 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %14 = llvm.extractvalue %arg1[1] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %15 = llvm.sub %14, %13 : !llvm.i64
// LLVM-NEXT: %16 = llvm.insertvalue %15, %12[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: %17 = llvm.extractvalue %arg1[2] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %18 = llvm.mul %5, %17 : !llvm.i64
// LLVM-NEXT: %19 = llvm.insertvalue %18, %16[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">

func @sliceRangeConversion(%arg0: memref<?x?xf32>, %arg1: !linalg.range, %arg2: !linalg.range, %arg3: !linalg.range) {
  %0 = linalg.view %arg0[%arg1, %arg2] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  %1 = linalg.slice %0[%arg3] {dim: 0} : !linalg.view<?x?xf32>, !linalg.range
  return
}
// LLVM-LABEL: @sliceRangeConversion
// LLVM:      %28 = llvm.undef : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %29 = llvm.extractvalue %27[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %30 = llvm.insertvalue %29, %28[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %31 = llvm.extractvalue %27[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %32 = llvm.extractvalue %arg3[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %33 = llvm.extractvalue %27[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %34 = llvm.mul %32, %33 : !llvm.i64
// LLVM-NEXT: %35 = llvm.add %31, %34 : !llvm.i64
// LLVM-NEXT: %36 = llvm.insertvalue %35, %30[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %37 = llvm.extractvalue %arg3[1] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %38 = llvm.extractvalue %arg3[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %39 = llvm.sub %37, %38 : !llvm.i64
// LLVM-NEXT: %40 = llvm.extractvalue %27[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %41 = llvm.extractvalue %arg3[2] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT: %42 = llvm.mul %40, %41 : !llvm.i64
// LLVM-NEXT: %43 = llvm.insertvalue %39, %36[2, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %44 = llvm.insertvalue %42, %43[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %45 = llvm.extractvalue %27[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %46 = llvm.extractvalue %27[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %47 = llvm.insertvalue %45, %44[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %48 = llvm.insertvalue %46, %47[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">

func @sliceNonRangeConversion2(%arg0: memref<?x?xf32>, %arg1: !linalg.range, %arg2: !linalg.range, %arg3: index) {
  %0 = linalg.view %arg0[%arg1, %arg2] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  %1 = linalg.slice %0[%arg3] {dim: 0} : !linalg.view<?x?xf32>, index
  return
}
// LLVM-LABEL: @sliceNonRangeConversion2
//      LLVM: %28 = llvm.undef : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: %29 = llvm.extractvalue %27[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %30 = llvm.insertvalue %29, %28[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: %31 = llvm.extractvalue %27[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %32 = llvm.extractvalue %27[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %33 = llvm.mul %arg3, %32 : !llvm.i64
// LLVM-NEXT: %34 = llvm.add %31, %33 : !llvm.i64
// LLVM-NEXT: %35 = llvm.insertvalue %34, %30[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: %36 = llvm.extractvalue %27[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %37 = llvm.extractvalue %27[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: %38 = llvm.insertvalue %36, %35[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: %39 = llvm.insertvalue %37, %38[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
