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
      %6 = linalg.slice %5[%i0] {dim = 1} : !linalg.view<?x?xf32>, index
      "some_consumer"(%6) : (!linalg.view<?xf32>) -> ()
      %7 = linalg.slice %5[%i1] {dim = 0} : !linalg.view<?x?xf32>, index
      %8 = linalg.slice %7[%i0] {dim = 0} : !linalg.view<?xf32>, index
    }
  }
  return
}
// CHECK-LABEL: func @slice_op(%{{.*}}: memref<?x?xf32>) {
//       CHECK:  %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32>
//       CHECK:  %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32>
//       CHECK:  %[[r1:.*]] = linalg.range %{{.*}}:%[[M]]:%{{.*}} : !linalg.range
//       CHECK:  %[[r2:.*]] = linalg.range %{{.*}}:%[[N]]:%{{.*}} : !linalg.range
//       CHECK:  %[[V:.*]] = linalg.view %{{.*}}[%[[r1]], %[[r2]]] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//       CHECK:  affine.for %{{.*}} = 0 to #map1(%{{.*}}) {
//       CHECK:   affine.for %{{.*}} = 0 to #map1(%{{.*}}) {
//       CHECK:     {{.*}} = linalg.slice %[[V]][%{{.*}}] {dim = 1} : !linalg.view<?x?xf32>, index
//       CHECK:     %[[V2:.*]] = linalg.slice %[[V]][%{{.*}}] {dim = 0} : !linalg.view<?x?xf32>, index
//       CHECK:     {{.*}} = linalg.slice %[[V2]][%{{.*}}] {dim = 0} : !linalg.view<?xf32>, index

func @rangeConversion(%arg0: index, %arg1: index, %arg2: index) {
  %0 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
  return
}
// LLVM-LABEL: @rangeConversion
// LLVM-NEXT:  llvm.mlir.undef : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">

func @viewRangeConversion(%arg0: memref<?x?xf32>, %arg1: !linalg.range, %arg2: !linalg.range) {
  %0 = linalg.view %arg0[%arg1, %arg2] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  return
}
// LLVM-LABEL: @viewRangeConversion
// LLVM-NEXT:  llvm.load %{{.*}} : !llvm<"{ float*, [2 x i64] }*">
// LLVM-NEXT:  llvm.mlir.undef : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, [2 x i64] }">
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[1, 1] : !llvm<"{ float*, [2 x i64] }">
// LLVM-NEXT:  llvm.mlir.constant(1 : index) : !llvm.i64
// LLVM-NEXT:  llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.mlir.constant(0 : index) : !llvm.i64
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.add %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.add %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.sub %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[2, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.sub %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">

func @viewNonRangeConversion(%arg0: memref<?x?xf32>, %arg1: !linalg.range, %arg2: index) {
  %0 = linalg.view %arg0[%arg1, %arg2] : memref<?x?xf32>, !linalg.range, index, !linalg.view<?xf32>
  return
}
// LLVM-LABEL: @viewNonRangeConversion
// LLVM-NEXT:  llvm.load %{{.*}} : !llvm<"{ float*, [2 x i64] }*">
// LLVM-NEXT:  llvm.mlir.undef : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, [2 x i64] }">
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[1, 1] : !llvm<"{ float*, [2 x i64] }">
// LLVM-NEXT:  llvm.mlir.constant(1 : index) : !llvm.i64
// LLVM-NEXT:  llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.mlir.constant(0 : index) : !llvm.i64
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.add %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.add %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.sub %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">

func @sliceRangeConversion(%arg0: memref<?x?xf32>, %arg1: !linalg.range, %arg2: !linalg.range, %arg3: !linalg.range) {
  %0 = linalg.view %arg0[%arg1, %arg2] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  %1 = linalg.slice %0[%arg3] {dim = 0} : !linalg.view<?x?xf32>, !linalg.range
  return
}
// LLVM-LABEL: @sliceRangeConversion
// LLVM:       llvm.mlir.undef : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM:       llvm.mlir.undef : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.add %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.sub %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
// LLVM-NEXT:  llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[2, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.extractvalue %{{.*}}[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">

func @sliceNonRangeConversion2(%arg0: memref<?x?xf32>, %arg1: !linalg.range, %arg2: !linalg.range, %arg3: index) {
  %0 = linalg.view %arg0[%arg1, %arg2] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  %1 = linalg.slice %0[%arg3] {dim = 0} : !linalg.view<?x?xf32>, index
  return
}
// LLVM-LABEL: @sliceNonRangeConversion2
//      LLVM: llvm.mlir.undef : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: llvm.extractvalue %{{.*}}[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: llvm.extractvalue %{{.*}}[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: llvm.mul %{{.*}}arg3, %{{.*}} : !llvm.i64
// LLVM-NEXT: llvm.add %{{.*}}, %{{.*}} : !llvm.i64
// LLVM-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: llvm.extractvalue %{{.*}}[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: llvm.extractvalue %{{.*}}[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
// LLVM-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
// LLVM-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
