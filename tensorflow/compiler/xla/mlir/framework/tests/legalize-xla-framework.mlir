// RUN: xla-translate-opt %s -xla-legalize-xla-framework-to-llvm | FileCheck %s

memref.global "private" constant @__constant_xf32 : memref<f32> = dense<42.0>

func.func @buffer_type(%arg: !xla_framework.buffer {xla_framework.input_mapping = 0 : i64})
                      attributes {xla_entry} {
  %val = xla_framework.buffer_to_mem %arg : memref<f32>
  %global = memref.get_global @__constant_xf32 : memref<f32>
  memref.copy %global, %val : memref<f32> to memref<f32>
  func.return
}

// CHECK-LABEL: @buffer_type
// The following signature is always the same.
// CHECK-SAME: %{{[^:]*}}: !llvm.ptr<i8>
// CHECK-SAME: %{{[^:]*}}: !llvm.ptr<i8>
// CHECK-SAME: %{{[^:]*}}: !llvm.ptr<ptr<i8>>
// CHECK-SAME: %[[BUFFERS:.*]]: !llvm.ptr<ptr<i8>>
// CHECK-SAME: %{{[^:]*}}: !llvm.ptr<i64>
// CHECK-SAME: %{{[^:]*}}: !llvm.ptr<i64>) {
// Retrieve pointer from the input as part of the function signature lowering.
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: %[[PTRS:.*]] = llvm.getelementptr %[[BUFFERS]][%[[C0]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK: %[[PTR0:.*]] = llvm.load %[[PTRS]] : !llvm.ptr<ptr<i8>>
// CHECK: %[[INP0:.*]] = llvm.bitcast %[[PTR0]] : !llvm.ptr<i8> to !llvm.ptr<f32>
// Create memref descriptor as the buffer_to_mem lowering.
// CHECK: %[[MEMREF:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK: %[[MEMREF1:.*]] = llvm.insertvalue %[[INP0]], %[[MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK: %[[MEMREF:.*]] = llvm.insertvalue %[[INP0]], %[[MEMREF1]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK: %[[C0_0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: llvm.insertvalue %[[C0_0:.*]], %[[MEMREF:.*]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// No return values in this case
// CHECK: return


func.func @return_tuple(%result0: !xla_framework.buffer, %result1: !xla_framework.buffer)
                      attributes {xla_entry, xla_framework.result_inner_mapping=[1,2], xla_framework.result_mapping=0} {
  func.return
}


// CHECK-LABEL: @return_tuple
// The following signature is always the same.
// CHECK-SAME: %{{[^:]*}}: !llvm.ptr<i8>
// CHECK-SAME: %{{[^:]*}}: !llvm.ptr<i8>
// CHECK-SAME: %{{[^:]*}}: !llvm.ptr<ptr<i8>>
// CHECK-SAME: %[[BUFFERS:.*]]: !llvm.ptr<ptr<i8>>
// CHECK-SAME: %{{[^:]*}}: !llvm.ptr<i64>
// CHECK-SAME: %{{[^:]*}}: !llvm.ptr<i64>) {
// Get Tuple
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK-NEXT: %[[PTRS0:.*]] = llvm.getelementptr %[[BUFFERS]][%[[C0]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK-NEXT: %[[PTR0:.*]] = llvm.load %[[PTRS0]] : !llvm.ptr<ptr<i8>>
// Get individual output buffer
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[PTRS1:.*]] = llvm.getelementptr %[[BUFFERS]][%[[C1]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK-NEXT: %[[PTR1:.*]] = llvm.load %[[PTRS1]] : !llvm.ptr<ptr<i8>>
// Store into tuple
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.bitcast %[[PTR0]] : !llvm.ptr<i8> to !llvm.ptr<ptr<i8>>
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[TUPLE_ELEMENT:.*]] = llvm.getelementptr %[[TUPLE]][%[[C0]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK-NEXT: llvm.store %[[PTR1]], %[[TUPLE_ELEMENT]] : !llvm.ptr<ptr<i8>>
// Get tuple
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK-NEXT: %[[PTRS0:.*]] = llvm.getelementptr %[[BUFFERS]][%[[C0]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK-NEXT: %[[PTR0:.*]] = llvm.load %[[PTRS0]] : !llvm.ptr<ptr<i8>>
// Get individual output buffer
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[PTRS2:.*]] = llvm.getelementptr %[[BUFFERS]][%[[C2]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK-NEXT: %[[PTR2:.*]] = llvm.load %[[PTRS2]] : !llvm.ptr<ptr<i8>>
// Store into Tuple
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.bitcast %[[PTR0]] : !llvm.ptr<i8> to !llvm.ptr<ptr<i8>>
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[TUPLE_ELEMENT:.*]] = llvm.getelementptr %[[TUPLE]][%[[C1]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK-NEXT: llvm.store %[[PTR2]], %[[TUPLE_ELEMENT]] : !llvm.ptr<ptr<i8>>
// No return values
// CHECK-NEXT:  return
