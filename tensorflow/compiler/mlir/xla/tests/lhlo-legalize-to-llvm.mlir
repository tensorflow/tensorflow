// RUN: xla-opt %s --test-lhlo-legalize-to-llvm -split-input-file | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: func @static_memref_cast
func @static_memref_cast(%buf : memref<10x1x5xf32>) {
  %0 = xla_lhlo.static_memref_cast %buf
        : memref<10x1x5xf32> -> memref<10x5xf32, offset: 2, strides: [5, 1]>
  return
}
// CHECK: %[[INPUT_MEMREF_BLDR:.*]] = llvm.mlir.undef : [[DESCRIPTOR_TYPE_3D:!.*]]
// CHECK: llvm.insertvalue
// CHECK: %[[MEMREF_BLDR_0:.*]] = llvm.mlir.undef : [[DESCRIPTOR_TYPE_2D:!.*]]

// CHECK: %[[IN_PTR:.*]] = llvm.extractvalue %[[INPUT_MEMREF:.*]][0] : [[DESCRIPTOR_TYPE_3D]]
// CHECK: %[[PTR:.*]] = llvm.bitcast %[[IN_PTR]] : !llvm<"float*"> to !llvm<"float*">
// CHECK: %[[MEMREF_BLDR_1:.*]] = llvm.insertvalue %[[PTR]], %[[MEMREF_BLDR_0]][0] : [[DESCRIPTOR_TYPE_2D]]

// CHECK: %[[IN_ALIGNED_PTR:.*]] = llvm.extractvalue %[[INPUT_MEMREF]][1] : [[DESCRIPTOR_TYPE_3D]]
// CHECK: %[[ALIGNED_PTR:.*]] = llvm.bitcast %[[IN_ALIGNED_PTR]] : !llvm<"float*"> to !llvm<"float*">
// CHECK: %[[MEMREF_BLDR_2:.*]] = llvm.insertvalue %[[ALIGNED_PTR]], %[[MEMREF_BLDR_1]][1] : [[DESCRIPTOR_TYPE_2D]]

// CHECK: %[[C2:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK: %[[MEMREF_BLDR_3:.*]] = llvm.insertvalue %[[C2]], %[[MEMREF_BLDR_2]][2] : [[DESCRIPTOR_TYPE_2D]]

// CHECK: %[[C10:.*]] = llvm.mlir.constant(10 : index) : !llvm.i64
// CHECK: %[[MEMREF_BLDR_4:.*]] = llvm.insertvalue %[[C10]], %[[MEMREF_BLDR_3]][3, 0] : [[DESCRIPTOR_TYPE_2D]]
// CHECK: %[[C5:.*]] = llvm.mlir.constant(5 : index) : !llvm.i64
// CHECK: %[[MEMREF_BLDR_5:.*]] = llvm.insertvalue %[[C5]], %[[MEMREF_BLDR_4]][4, 0] : [[DESCRIPTOR_TYPE_2D]]
// CHECK: %[[C5_:.*]] = llvm.mlir.constant(5 : index) : !llvm.i64
// CHECK: %[[MEMREF_BLDR_6:.*]] = llvm.insertvalue %[[C5_]], %[[MEMREF_BLDR_5]][3, 1] : [[DESCRIPTOR_TYPE_2D]]
// CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: %[[MEMREF_BLDR_7:.*]] = llvm.insertvalue %[[C1]], %[[MEMREF_BLDR_6]][4, 1] : [[DESCRIPTOR_TYPE_2D]]

// -----

// CHECK-LABEL: func @dynamic_memref_cast
func @dynamic_memref_cast(%buf : memref<?x?xf32>) {
  %size_X = constant 10 : index
  %size_Y = constant 50 : index
  %stride_X = constant 1 : index
  %stride_Y = constant 0 : index
  %0 = xla_lhlo.dynamic_memref_cast %buf(%size_X, %size_Y)[%stride_X, %stride_Y]
        : memref<?x?xf32> -> memref<?x?xf32, offset: 0, strides: [?, ?]>
  return
}
// CHECK: %[[C10:.*]] = llvm.mlir.constant(10 : index) : !llvm.i64
// CHECK: %[[C50:.*]] = llvm.mlir.constant(50 : index) : !llvm.i64
// CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64

// CHECK: %[[MEMREF_BLDR_0:.*]] = llvm.mlir.undef : [[DESCRIPTOR_TYPE:!.*]]

// CHECK: %[[IN_PTR:.*]] = llvm.extractvalue %[[INPUT_MEMREF:.*]][0] : [[DESCRIPTOR_TYPE]]
// CHECK: %[[PTR:.*]] = llvm.bitcast %[[IN_PTR]] : !llvm<"float*"> to !llvm<"float*">
// CHECK: %[[MEMREF_BLDR_1:.*]] = llvm.insertvalue %[[PTR]], %[[MEMREF_BLDR_0]][0] : [[DESCRIPTOR_TYPE]]

// CHECK: %[[IN_ALIGNED_PTR:.*]] = llvm.extractvalue %[[INPUT_MEMREF]][1] : [[DESCRIPTOR_TYPE]]
// CHECK: %[[ALIGNED_PTR:.*]] = llvm.bitcast %[[IN_ALIGNED_PTR]] : !llvm<"float*"> to !llvm<"float*">
// CHECK: %[[MEMREF_BLDR_2:.*]] = llvm.insertvalue %[[ALIGNED_PTR]], %[[MEMREF_BLDR_1]][1] : [[DESCRIPTOR_TYPE]]

// CHECK: %[[SRC_OFFSET:.*]] = llvm.extractvalue %[[INPUT_MEMREF]][2] : [[DESCRIPTOR_TYPE]]
// CHECK: %[[MEMREF_BLDR_3:.*]] = llvm.insertvalue %[[SRC_OFFSET]], %[[MEMREF_BLDR_2]][2] : [[DESCRIPTOR_TYPE]]
// CHECK: %[[MEMREF_BLDR_4:.*]] = llvm.insertvalue %[[C10]], %[[MEMREF_BLDR_3]][3, 0] : [[DESCRIPTOR_TYPE]]
// CHECK: %[[MEMREF_BLDR_5:.*]] = llvm.insertvalue %[[C1]], %[[MEMREF_BLDR_4]][4, 0] : [[DESCRIPTOR_TYPE]]
// CHECK: %[[MEMREF_BLDR_6:.*]] = llvm.insertvalue %[[C50]], %[[MEMREF_BLDR_5]][3, 1] : [[DESCRIPTOR_TYPE]]
// CHECK: %[[MEMREF_BLDR_7:.*]] = llvm.insertvalue %[[C0]], %[[MEMREF_BLDR_6]][4, 1] : [[DESCRIPTOR_TYPE]]
