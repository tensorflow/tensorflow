// RUN: mlir-opt %s -convert-linalg-to-llvm | FileCheck %s
// RUN: mlir-opt %s -linalg-lower-to-loops -convert-linalg-to-llvm | FileCheck %s --check-prefix=LLVM-LOOPS

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

func @slice(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: !linalg.range) {
  %1 = linalg.slice %arg0[%arg1] : memref<?xf32, offset: ?, strides: [1]>, !linalg.range, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @slice
//   insert data ptr for slice op
//       CHECK:   llvm.extractvalue %{{.*}}[4, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.add %{{.*}}, %{{.*}} : !llvm.i64
//    insert offset
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.mlir.constant(0 : index)
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
//    get size[0] from parent view
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
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
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">

func @dot(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
//      CHECK-LABEL: func @dot(%{{.*}}: !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">, %{{.*}}: !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">, %{{.*}}: !llvm<"{ float*, float*, i64 }*">) {
//    CHECK-COUNT-3:   llvm.mlir.constant(1 : index){{.*[[:space:]].*}}llvm.alloca{{.*[[:space:]].*}}llvm.store
//       CHECK-NEXT:   llvm.call @linalg_dot_viewsxf32_viewsxf32_viewf32(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">, !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">, !llvm<"{ float*, float*, i64 }*">) -> ()

func @slice_with_range_and_index(%arg0: memref<?x?xf64, offset: ?, strides: [?, 1]>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %R = linalg.range %c0:%c1:%c1 : !linalg.range
  loop.for %i0 = %c0 to %c1 step %c1 {
    %1 = linalg.slice %arg0[%i0, %R] : memref<?x?xf64, offset: ?, strides: [?, 1]>, index, !linalg.range, memref<?xf64, offset: ?, strides: [1]>
  }
  return
}
// CHECK-LABEL: func @slice_with_range_and_index
// loop-body.
//       CHECK:   llvm.mlir.undef : !llvm<"{ double*, double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[4, 0] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[4, 1] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ double*, double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}[2] : !llvm<"{ double*, double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   llvm.insertvalue %{{.*}}[3, 0] : !llvm<"{ double*, double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}[4, 0] : !llvm<"{ double*, double*, i64, [1 x i64], [1 x i64] }">

func @copy(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy
//       CHECK:   llvm.call @linalg_copy_viewsxsxsxf32_viewsxsxsxf32(%{{.*}}, %{{.*}}) : (!llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*">, !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*">) -> ()

func @transpose(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  %0 = linalg.transpose %arg0 (i, j, k) -> (k, i, j) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @transpose
//       CHECK:   llvm.mlir.undef : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.extractvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.extractvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.extractvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">

func @copy_transpose(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = (i, j, k) -> (i, k, j),
                             outputPermutation = (i, j, k) -> (k, j, i)}
    : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy
// Tranpose input
//         CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
// Transpose output
//         CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
// Call external copy after promoting input and output structs to pointers
// CHECK-COUNT-2:   llvm.mlir.constant(1 : index){{.*[[:space:]].*}}llvm.alloca{{.*[[:space:]].*}}llvm.store
//         CHECK:   llvm.call @linalg_copy_viewsxsxsxf32_viewsxsxsxf32(%{{.*}}, %{{.*}}) : (!llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*">, !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*">) -> ()

#matmul_accesses = [
  (m, n, k) -> (m, k),
  (m, n, k) -> (k, n),
  (m, n, k) -> (m, n)
]
#matmul_trait = {
  n_views = [2, 1],
  n_loop_types = [2, 1, 0],
  indexing_maps = #matmul_accesses,
  library_call = "some_external_function_name_for_vector_outerproduct_matmul"
}

!vector_type_A = type vector<4xf32>
!vector_type_B = type vector<4xf32>
!vector_type_C = type vector<4x4xf32>

!matrix_type_A = type memref<?x?x!vector_type_A>
!matrix_type_B = type memref<?x?x!vector_type_B>
!matrix_type_C = type memref<?x?x!vector_type_C>

func @matmul_vec_impl(%A: !matrix_type_A, %B: !matrix_type_B, %C: !matrix_type_C) {
  linalg.generic #matmul_trait %A, %B, %C {
    ^bb0(%a: !vector_type_A, %b: !vector_type_B, %c: !vector_type_C):
      %d = vector.outerproduct %a, %b, %c: !vector_type_A, !vector_type_B
      linalg.yield %d: !vector_type_C
  } : !matrix_type_A, !matrix_type_B, !matrix_type_C

  return
}
// CHECK-LABEL: func @matmul_vec_impl(
//   CHECK:  llvm.call @some_external_function_name_for_vector_outerproduct_matmul(%{{.*}}) : (!llvm<"{ <4 x float>*, <4 x float>*, i64, [2 x i64], [2 x i64] }*">, !llvm<"{ <4 x float>*, <4 x float>*, i64, [2 x i64], [2 x i64] }*">, !llvm<"{ [4 x <4 x float>]*, [4 x <4 x float>]*, i64, [2 x i64], [2 x i64] }*">) -> ()

// LLVM-LOOPS-LABEL: func @matmul_vec_impl(
//   LLVM-LOOPS: llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
//   LLVM-LOOPS: llvm.shufflevector {{.*}} [1 : i32, 1 : i32, 1 : i32, 1 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
//   LLVM-LOOPS: llvm.shufflevector {{.*}} [2 : i32, 2 : i32, 2 : i32, 2 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
//   LLVM-LOOPS: llvm.shufflevector {{.*}} [3 : i32, 3 : i32, 3 : i32, 3 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
//   LLVM-LOOPS-NEXT: llvm.extractvalue {{.*}}[3] : !llvm<"[4 x <4 x float>]">
//   LLVM-LOOPS-NEXT: "llvm.intr.fmuladd"({{.*}}) : (!llvm<"<4 x float>">, !llvm<"<4 x float>">, !llvm<"<4 x float>">) -> !llvm<"<4 x float>">
//   LLVM-LOOPS-NEXT: llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x <4 x float>]">
