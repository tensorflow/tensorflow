// RUN: mlir-hlo-opt -hlo-convert-deallocation-ops-to-llvm %s \
// RUN: -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @unranked_null()
func.func @unranked_null() {
  %null = deallocation.null : memref<*xf32>
  func.return
}
// CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: [[DESC_0:%.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
// CHECK: [[DESC_1:%.*]] = llvm.insertvalue [[C0]], [[DESC_0]][0]
// CHECK: [[PTR:%.*]] = llvm.alloca {{.*}} x i8
// CHECK: [[NULL:%.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK: [[BITCAST:%.*]] = llvm.bitcast [[PTR]] : !llvm.ptr<i8> to !llvm.ptr<ptr<f32>>
// CHECK: llvm.store [[NULL]], [[BITCAST]] : !llvm.ptr<ptr<f32>>
// CHECK: [[DESC_2:%.*]] = llvm.insertvalue [[PTR]], [[DESC_1]][1]

// -----

// CHECK-LABEL: func.func @ranked_null()
func.func @ranked_null() {
  %null = deallocation.null : memref<2x?xf32>
  func.return
}
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[C1_:.*]] = llvm.mlir.constant(1 : index) : i64

// CHECK: llvm.mlir.null
// CHECK: %[[NULL:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK-NEXT: %[[DESC_0:.*]] = llvm.mlir.undef :
// CHECK-SAME:   !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[DESC_1:.*]] = llvm.insertvalue %[[NULL]], %[[DESC_0]][0]
// CHECK-NEXT: %[[DESC_2:.*]] = llvm.insertvalue %[[NULL]], %[[DESC_1]][1]
// CHECK-NEXT: %[[DESC_3:.*]] = llvm.insertvalue %[[C0]], %[[DESC_2]][2]
// CHECK-NEXT: %[[DESC_4:.*]] = llvm.insertvalue %[[C2]], %[[DESC_3]][3, 0]
// CHECK-NEXT: %[[DESC_5:.*]] = llvm.insertvalue %[[C1]], %[[DESC_4]][4, 0]
// CHECK-NEXT: %[[DESC_6:.*]] = llvm.insertvalue %[[C1]], %[[DESC_5]][3, 1]
// CHECK-NEXT: %[[DESC_7:.*]] = llvm.insertvalue %[[C1_]], %[[DESC_6]][4, 1]

// -----

// CHECK-LABEL: func.func @unranked_get_buffer
func.func @unranked_get_buffer(%arg0: memref<*xf32>) -> index {
  %ret = deallocation.get_buffer %arg0 : memref<*xf32>
  return %ret : index
}

// CHECK-NEXT: builtin.unrealized_conversion_cast
// CHECK-NEXT: llvm.extractvalue
// CHECK-NEXT: llvm.bitcast
// CHECK-NEXT: llvm.load
// CHECK-NEXT: llvm.ptrtoint

// -----

// CHECK-LABEL: func.func @ranked_get_buffer
func.func @ranked_get_buffer(%arg0: memref<2x?xf32>) -> index {
  %ret = deallocation.get_buffer %arg0 : memref<2x?xf32>
  return %ret : index
}

// CHECK-NEXT: builtin.unrealized_conversion_cast
// CHECK-NEXT: llvm.extractvalue
// CHECK-NEXT: llvm.ptrtoint
