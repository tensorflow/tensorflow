// RUN: xla-cpu-opt -xla-convert-memref-element-cast-to-llvm %s \
// RUN: -split-input-file | FileCheck %s

func.func @memref_cast(%arg0: memref<10xf32>) -> memref<10xi32> {
  %ret = xla_cpu.memref_element_cast %arg0 : memref<10xf32> to memref<10xi32>
  return %ret : memref<10xi32>
}
// CHECK-LABEL: func.func @memref_cast(
// CHECK-SAME:      %[[SRC:.*]]: memref<10xf32>) -> memref<10xi32>
// CHECK:         %[[SRC_DESC:.*]] = builtin.unrealized_conversion_cast %[[SRC]]
// CHECK-SAME:      : memref<10xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[ALLOC_PTR:.*]] = llvm.extractvalue %[[SRC_DESC]][0]
// CHECK-NEXT:    %[[ALIGN_PTR:.*]] = llvm.extractvalue %[[SRC_DESC]][1]

// CHECK:         %[[ALLOC_PTR_CAST:.*]] = llvm.bitcast %[[ALLOC_PTR]] : !llvm.ptr<f32> to !llvm.ptr<i32>
// CHECK-NEXT:    %[[ALIGN_PTR_CAST:.*]] = llvm.bitcast %[[ALIGN_PTR]] : !llvm.ptr<f32> to !llvm.ptr<i32>

// CHECK:         %[[DST_DESC:.*]] = llvm.mlir.undef
// CHECK-SAME:      : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[DST_DESC_:.*]] = llvm.insertvalue %[[ALLOC_PTR_CAST]], %[[DST_DESC]][0]
// CHECK-NEXT:    llvm.insertvalue %[[ALIGN_PTR_CAST]], %[[DST_DESC_]][1]

// CHECK:         builtin.unrealized_conversion_cast
// CHECK-SAME:      : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)> to memref<10xi32>

//  -----

func.func @memref_cast_i1(%arg0: memref<10xi1>) -> memref<10xi8> {
  %ret = xla_cpu.memref_element_cast %arg0 : memref<10xi1> to memref<10xi8>
  return %ret : memref<10xi8>
}
// CHECK-LABEL: func.func @memref_cast_i1(
// CHECK-SAME:      %[[SRC:.*]]: memref<10xi1>) -> memref<10xi8>
// CHECK:         %[[SRC_DESC:.*]] = builtin.unrealized_conversion_cast %[[SRC]]
// CHECK-SAME:      : memref<10xi1> to !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[ALLOC_PTR:.*]] = llvm.extractvalue %[[SRC_DESC]][0]
// CHECK-NEXT:    %[[ALIGN_PTR:.*]] = llvm.extractvalue %[[SRC_DESC]][1]

// CHECK:         %[[ALLOC_PTR_CAST:.*]] = llvm.bitcast %[[ALLOC_PTR]] : !llvm.ptr<i1> to !llvm.ptr<i8>
// CHECK-NEXT:    %[[ALIGN_PTR_CAST:.*]] = llvm.bitcast %[[ALIGN_PTR]] : !llvm.ptr<i1> to !llvm.ptr<i8>

// CHECK:         %[[DST_DESC:.*]] = llvm.mlir.undef
// CHECK-SAME:      : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[DST_DESC_:.*]] = llvm.insertvalue %[[ALLOC_PTR_CAST]], %[[DST_DESC]][0]
// CHECK-NEXT:    llvm.insertvalue %[[ALIGN_PTR_CAST]], %[[DST_DESC_]][1]

// CHECK:         builtin.unrealized_conversion_cast
// CHECK-SAME:      : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)> to memref<10xi8>
