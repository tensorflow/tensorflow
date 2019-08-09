// RUN: mlir-opt %s -linalg-lower-to-llvm-dialect | FileCheck %s

func @buffer_size(%arg0: !linalg.buffer<?xf32>) {
  %c1 = constant 1 : index
  %s = linalg.buffer_size %arg0 : !linalg.buffer<?xf32>
  %t = addi %s, %c1 : index
  return
}
// CHECK-LABEL: func @buffer_size(%{{.*}}: !llvm<"{ float*, i64 }">) {
//       CHECK:   %{{.*}} = llvm.extractvalue %{{.*}}[1] : !llvm<"{ float*, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.add {{.*}}, {{.*}} : !llvm.i64

func @range(%arg0: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %R = linalg.range %c0:%arg0:%c1 : !linalg.range
  return
}
// CHECK-LABEL: func @range(%{{.*}}: !llvm.i64) {
//       CHECK:   %{{.*}} = llvm.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.undef : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">

func @view(%arg0: !linalg.buffer<?xf32>, %arg1: !linalg.range) {
  %0 = linalg.view %arg0[%arg1] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
  return
}
// CHECK-LABEL: func @view(%{{.*}}: !llvm<"{ float*, i64 }">, %{{.*}}: !llvm<"{ i64, i64, i64 }">) {
//       CHECK:   %{{.*}} = llvm.undef : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %{{.*}} = llvm.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %{{.*}} = llvm.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.sub %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">

func @view3d(%arg0: !linalg.buffer<?xf32>, %arg1: !linalg.range, %arg2: !linalg.range, %arg3: !linalg.range) {
  %0 = linalg.view %arg0[%arg1, %arg2, %arg3] : !linalg.buffer<?xf32> -> !linalg.view<?x?x?xf32>
  return
}
// CHECK-LABEL: func @view3d(%{{.*}}: !llvm<"{ float*, i64 }">, %{{.*}}: !llvm<"{ i64, i64, i64 }">, %{{.*}}: !llvm<"{ i64, i64, i64 }">, %{{.*}}: !llvm<"{ i64, i64, i64 }">) {
//       CHECK:   %{{.*}} = llvm.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[3, 2] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   %{{.*}} = llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">

func @slice(%arg0: !linalg.buffer<?xf32>, %arg1: !linalg.range) {
  %0 = linalg.view %arg0[%arg1] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
  %1 = linalg.slice %0[%arg1] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
  return
}
// CHECK-LABEL: func @slice(%{{.*}}: !llvm<"{ float*, i64 }">, %{{.*}}: !llvm<"{ i64, i64, i64 }">) {
//       CHECK:   %{{.*}} = llvm.undef : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   %{{.*}} = llvm.undef : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.add %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.sub %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   %{{.*}} = llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">

func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// CHECK-LABEL: func @dot(%{{.*}}: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, %{{.*}}: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, %{{.*}}: !llvm<"{ float*, i64, [0 x i64], [0 x i64] }">) {
//       CHECK:   llvm.call @linalg_dot_viewxf32_viewxf32_viewf32(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">, !llvm<"{ float*, i64, [0 x i64], [0 x i64] }">) -> ()

func @dim(%arg0: !linalg.view<?x?xf32>) {
  %0 = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
  return
}
// CHECK-LABEL: func @dim(%{{.*}}: !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">) {
//       CHECK:   %{{.*}} = llvm.extractvalue %{{.*}}[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">

func @subview(%arg0: !linalg.view<?x?xf32>) {
  %c0 = constant 0 : index
  %0 = linalg.subview %arg0[%c0, %c0, %c0, %c0, %c0, %c0] : !linalg.view<?x?xf32>
  return
}
// CHECK-LABEL: func @subview(%{{.*}}: !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">) {
//       CHECK:   %{{.*}} = llvm.constant(0 : index) : !llvm.i64
//       CHECK:   %{{.*}} = llvm.extractvalue %{{.*}}[2, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   %{{.*}} = llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm.i64
//       CHECK:   %{{.*}} = llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.i64
//       CHECK:   %{{.*}} = llvm.undef : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %{{.*}} = llvm.extractvalue %{{.*}}[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   %{{.*}} = llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm.i64
//       CHECK:   %{{.*}} = llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.i64
//       CHECK:   %{{.*}} = llvm.undef : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
