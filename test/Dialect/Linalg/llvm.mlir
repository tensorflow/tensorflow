// RUN: mlir-opt %s -linalg-convert-to-llvm | FileCheck %s

func @buffer_size(%arg0: !linalg.buffer<?xf32>) {
  %c1 = constant 1 : index
  %s = linalg.buffer_size %arg0 : !linalg.buffer<?xf32>
  %t = addi %s, %c1 : index
  return
}
// CHECK-LABEL: func @buffer_size
//       CHECK:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ i8*, float*, i64 }">
//  CHECK-NEXT:   llvm.add {{.*}}, {{.*}} : !llvm.i64

func @buffer_alloc_aligned(%arg0: index) {
  %s = linalg.buffer_alloc %arg0 {alignment=16} : !linalg.buffer<?xf32>
  return
}
// CHECK-LABEL: func @buffer_alloc_aligned
//       CHECK:    %[[c4:.*]] = llvm.mlir.constant(4 : index) : !llvm.i64
//       CHECK:    %[[m:.*]] = llvm.mul %arg0, %[[c4]] : !llvm.i64
//       CHECK:    %[[c1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:    %[[c16:.*]] = llvm.mlir.constant(16 : index) : !llvm.i64
//       CHECK:    %[[a:.*]] = llvm.add %[[m]], %[[c16]] : !llvm.i64
//       CHECK:    %[[s:.*]] = llvm.sub %[[a]], %[[c1]] : !llvm.i64
//       CHECK:    %[[alloc:.*]] = llvm.call @malloc(%[[s]]) : (!llvm.i64) -> !llvm<"i8*">
// aligning `ptr` on `align` is done computing the address `ptr + (align - ptr % align) % align`.
//       CHECK:    %[[cast:.*]] = llvm.ptrtoint %[[alloc]] : !llvm<"i8*"> to !llvm.i64
//       CHECK:    %[[rem:.*]] = llvm.urem %[[cast]], %[[c16]] : !llvm.i64
//       CHECK:    %[[drem:.*]] = llvm.sub %[[c16]], %[[rem]] : !llvm.i64
//       CHECK:    %[[off:.*]] = llvm.urem %[[drem]], %[[c16]] : !llvm.i64
//       CHECK:    llvm.getelementptr %{{.*}}[%[[off]]] : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">

func @range(%arg0: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %R = linalg.range %c0:%arg0:%c1 : !linalg.range
  return
}
// CHECK-LABEL: func @range(%{{.*}}: !llvm.i64) {
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:   llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:   llvm.mlir.undef : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">

func @view(%arg0: !linalg.buffer<?xf32>, %arg1: !linalg.range) {
  %0 = linalg.view %arg0[%arg1] : !linalg.buffer<?xf32> -> memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @view
//  CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ i8*, float*, i64 }">
//  CHECK-NEXT:   llvm.bitcast {{.*}} : !llvm<"float*"> to !llvm<"float*">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.sub %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.return

func @view3d(%arg0: !linalg.buffer<?xf32>, %arg1: !linalg.range, %arg2: !linalg.range, %arg3: !linalg.range) {
  %0 = linalg.view %arg0[%arg1, %arg2, %arg3] : !linalg.buffer<?xf32> -> memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @view3d
//       CHECK:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 2] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">

func @slice(%arg0: !linalg.buffer<?xf32>, %arg1: !linalg.range) {
  %0 = linalg.view %arg0[%arg1] : !linalg.buffer<?xf32> -> memref<?xf32, offset: ?, strides: [1]>
  %1 = linalg.slice %0[%arg1] : memref<?xf32, offset: ?, strides: [1]>, !linalg.range, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @slice
//   insert ptr for view op
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//   insert data ptr for slice op
//       CHECK:   llvm.extractvalue %{{.*}}[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.add %{{.*}}, %{{.*}} : !llvm.i64
//    insert offset
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.mlir.constant(0 : index)
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
//    get size[0] from parent view
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.i64
//    compute size[0] bounded by parent view's size[0]
//  CHECK-NEXT:   llvm.sub %{{.*}}, %{{.*}} : !llvm.i64
//    bound below by 0
//  CHECK-NEXT:   llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.i64
//    compute stride[0] using bounded size
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//    insert size and stride
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">

func @dot(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
//      CHECK-LABEL: func @dot(%{{.*}}: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">, %{{.*}}: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">, %{{.*}}: !llvm<"{ float*, i64 }*">) {
//    CHECK-COUNT-3:   llvm.mlir.constant(1 : index){{.*[[:space:]].*}}llvm.alloca{{.*[[:space:]].*}}llvm.store
//       CHECK-NEXT:   llvm.call @linalg_dot_viewsxf32_viewsxf32_viewf32(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">, !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">, !llvm<"{ float*, i64 }*">) -> ()

func @dim(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  %0 = dim %arg0, 1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// CHECK-LABEL: func @dim(%{{.*}}: !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">) {
//       CHECK:   llvm.extractvalue %{{.*}}[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">

func @subview(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  %c0 = constant 0 : index
  %0 = linalg.subview %arg0[%c0, %c0, %c0, %c0, %c0, %c0] : memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// CHECK-LABEL: func @subview
//
// Subview lowers to range + slice op
//       CHECK:   llvm.mlir.undef : !llvm<"{ i64, i64, i64 }">
//       CHECK:   llvm.mlir.undef : !llvm<"{ i64, i64, i64 }">
//       CHECK:   llvm.mlir.undef : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
//
// Select occurs in slice op lowering
//       CHECK:   llvm.extractvalue %{{.*}}[2, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:   llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.i64
//  CHECK-NEXT:   llvm.sub %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.i64
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//
//       CHECK:   llvm.extractvalue %{{.*}}[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:   llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.i64
//  CHECK-NEXT:   llvm.sub %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.i64
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : !llvm.i64

func @view_with_range_and_index(%arg0: memref<?x?xf64, offset: ?, strides: [?, 1]>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %R = linalg.range %c0:%c1:%c1 : !linalg.range
  loop.for %i0 = %c0 to %c1 step %c1 {
    %1 = linalg.slice %arg0[%i0, %R] : memref<?x?xf64, offset: ?, strides: [?, 1]>, index, !linalg.range, memref<?xf64, offset: ?, strides: [1]>
  }
  return
}
// CHECK-LABEL: func @view_with_range_and_index
// loop-body.
//       CHECK:   llvm.mlir.undef : !llvm<"{ double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[3, 0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[3, 1] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}[1] : !llvm<"{ double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   llvm.insertvalue %{{.*}}[2, 0] : !llvm<"{ double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}[3, 0] : !llvm<"{ double*, i64, [1 x i64], [1 x i64] }">

func @copy(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy
//       CHECK:   llvm.call @linalg_copy_viewsxsxsxf32_viewsxsxsxf32(%{{.*}}, %{{.*}}) : (!llvm<"{ float*, i64, [3 x i64], [3 x i64] }*">, !llvm<"{ float*, i64, [3 x i64], [3 x i64] }*">) -> ()

func @transpose(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  %0 = linalg.transpose %arg0 (i, j, k) -> (k, i, j) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @transpose
//       CHECK:   llvm.mlir.undef : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.extractvalue {{.*}}[2, 0] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[2, 2] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.extractvalue {{.*}}[2, 1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[2, 0] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.extractvalue {{.*}}[2, 2] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[2, 1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">

func @copy_transpose(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = (i, j, k) -> (i, k, j),
                             outputPermutation = (i, j, k) -> (k, j, i)}
    : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy
// Tranpose input
//         CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[2, 0] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[2, 0] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[2, 1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[2, 2] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[2, 2] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[2, 1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
// Transpose output
//         CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[2, 0] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[2, 2] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[2, 1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[2, 1] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[2, 2] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[2, 0] : !llvm<"{ float*, i64, [3 x i64], [3 x i64] }">
// Call external copy after promoting input and output structs to pointers
// CHECK-COUNT-2:   llvm.mlir.constant(1 : index){{.*[[:space:]].*}}llvm.alloca{{.*[[:space:]].*}}llvm.store
//         CHECK:   llvm.call @linalg_copy_viewsxsxsxf32_viewsxsxsxf32(%{{.*}}, %{{.*}}) : (!llvm<"{ float*, i64, [3 x i64], [3 x i64] }*">, !llvm<"{ float*, i64, [3 x i64], [3 x i64] }*">) -> ()
