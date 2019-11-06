
// CHECK-LABEL: func @view(
// CHECK: %[[ARG0:.*]]: !llvm.i64, %[[ARG1:.*]]: !llvm.i64, %[[ARG2:.*]]: !llvm.i64
func @view(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: llvm.mlir.constant(2048 : index) : !llvm.i64
  // CHECK: llvm.mlir.undef : !llvm<"{ i8*, i64, [1 x i64], [1 x i64] }">
  %0 = alloc() : memref<2048xi8>

  // Test one dynamic size and dynamic offset.
  // CHECK: llvm.mlir.undef : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm<"{ i8*, i64, [1 x i64], [1 x i64] }">
  // CHECK: llvm.bitcast %{{.*}} : !llvm<"i8*"> to !llvm<"float*">
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: llvm.insertvalue %[[ARG2]], %{{.*}}[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: llvm.insertvalue %[[ARG1]], %{{.*}}[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: llvm.mlir.constant(1 : index) : !llvm.i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: llvm.mlir.constant(4 : index) : !llvm.i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[2, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: llvm.mlir.constant(4 : index) : !llvm.i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  %3 = view %0[%arg1][%arg2]
    : memref<2048xi8> to memref<4x?xf32, (d0, d1)[s0, s1] -> (d0 * s0 + d1 + s1)>
  return
}
