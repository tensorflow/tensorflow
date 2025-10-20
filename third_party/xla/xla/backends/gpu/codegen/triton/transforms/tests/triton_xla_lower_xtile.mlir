// RUN: xla-opt %s -split-input-file -triton-xla-lower-xtile | FileCheck %s

xtile.entry_func @extract_insert_no_layout(%input: memref<1024xf32, #nvvm.memory_space<global>>,
                         %output: memref<32xf32, #nvvm.memory_space<global>>,
                         %tile_id: index) {
  %tile = xtile.extract %input[%tile_id][1][1] : memref<1024xf32, #nvvm.memory_space<global>> -> tensor<1xf32>
  xtile.insert %tile into %output[%tile_id][1][1] : tensor<1xf32> -> memref<32xf32, #nvvm.memory_space<global>>
  xtile.return
}

// CHECK: func.func @extract_insert_no_layout(%[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<f32>) {
// CHECK:   %[[PID:.*]] = tt.get_program_id x : i32
// CHECK:   %[[PID_I64:.*]] = arith.extsi %[[PID]] : i32 to i64
// CHECK:   %[[PID_IDX:.*]] = arith.index_cast %[[PID_I64]] : i64 to index
// CHECK:   %[[TILE:.*]] = triton_xla.extract from %[[ARG0]] as memref<1024xf32, #triton_xla.layout<[0]>> [%[[PID_IDX]]] [1] [1] : tensor<1xf32>
// CHECK:   triton_xla.insert %[[TILE]] into %[[ARG1]] as memref<32xf32, #triton_xla.layout<[0]>> [%[[PID_IDX]]] [1] [1] : tensor<1xf32>
// CHECK:   return
// CHECK: }

// -----

!arg_type = memref<1024x32x1x1xbf16, #triton_xla.layout<[2, 3, 0, 1]>, #nvvm.memory_space<global>>
xtile.entry_func @layout_preserved(%input: !arg_type,
                                   %tile_id: index) {
  %c_0 = arith.constant 0 : index
  %tile = xtile.extract %input[%tile_id, %c_0, %c_0, %c_0][1, 1, 1, 1][1, 1, 1, 1] : !arg_type -> tensor<1x1x1x1xbf16>
  xtile.return
}

// CHECK: func.func @layout_preserved(%[[ARG0:.*]]: !tt.ptr<bf16>) {
// CHECK:   %[[PID:.*]] = tt.get_program_id x : i32
// CHECK:   %[[PID_I64:.*]] = arith.extsi %[[PID]] : i32 to i64
// CHECK:   %[[PID_IDX:.*]] = arith.index_cast %[[PID_I64]] : i64 to index
// CHECK:   %[[TILE:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME: as memref<1024x32x1x1xbf16, #triton_xla.layout<[3, 2, 0, 1]>>
// CHECK-SAME: [%[[PID_IDX]], 0, 0, 0]
// CHECK-SAME: [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xbf16>
// CHECK:   return
// CHECK: }

// -----

!memref_type = memref<32xf64, #nvvm.memory_space<global>>
// CHECK:func.func @scalar_insert_extract(
// CHECK-SAME: %[[ARG0:.*]]: !tt.ptr<f64>, %[[ARG1:.*]]: !tt.ptr<f64>) {
xtile.entry_func @scalar_insert_extract(%input: !memref_type,
                                        %output: !memref_type,
                                        %tile_id: index) {
  // CHECK: %[[SCALAR_VALUE:.*]] = tt.load %[[ARG0]] : !tt.ptr<f64>
  %tile = xtile.extract %input[%tile_id][1][1] : !memref_type -> tensor<f64>
  // CHECK: tt.store %[[ARG1]], %[[SCALAR_VALUE]] : !tt.ptr<f64>
  xtile.insert %tile into %output[%tile_id][1][1] : tensor<f64> -> !memref_type
  xtile.return
}

// -----

// CHECK-LABEL: func.func @fold_transpose_into_ptr
// CHECK-SAME: (%[[INPUT:.*]]: memref<32x16xf64, #triton_xla.layout<[0, 1]>>)
func.func @fold_transpose_into_ptr(
    %input: memref<32x16xf64, #triton_xla.layout<[0, 1]>>) -> !tt.ptr<f64> {
  %transposed = memref.transpose %input (d0, d1) -> (d1, d0)
    : memref<32x16xf64, #triton_xla.layout<[0, 1]>> to memref<16x32xf64>
  // CHECK: %[[PTR:.*]] = triton_xla.memref_to_ptr %[[INPUT]] from memref<32x16xf64, #triton_xla.layout<[0, 1]>> to <f64>
  %ptr = triton_xla.memref_to_ptr %transposed from memref<16x32xf64> to !tt.ptr<f64>
  // CHECK: return %[[PTR]] : !tt.ptr<f64>
  return %ptr : !tt.ptr<f64>
}
