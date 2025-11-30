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
// CHECK:   %[[PID_IDX:.*]] = arith.index_cast %[[PID]] : i32 to index
// CHECK:   %[[TILE:.*]] = triton_xla.extract from %[[ARG0]] as memref<1024xf32, #xtile.layout<[0]>> [%[[PID_IDX]]] [1] [1] : tensor<1xf32>
// CHECK:   triton_xla.insert %[[TILE]] into %[[ARG1]] as memref<32xf32, #xtile.layout<[0]>> [%[[PID_IDX]]] [1] [1] : tensor<1xf32>
// CHECK:   return
// CHECK: }

// -----

!arg_type = memref<1024x32x1x1xbf16, #xtile.layout<[2, 3, 0, 1]>, #nvvm.memory_space<global>>
xtile.entry_func @layout_preserved(%input: !arg_type,
                                   %tile_id: index) {
  %c_0 = arith.constant 0 : index
  %tile = xtile.extract %input[%tile_id, %c_0, %c_0, %c_0][1, 1, 1, 1][1, 1, 1, 1] : !arg_type -> tensor<1x1x1x1xbf16>
  xtile.return
}

// CHECK: func.func @layout_preserved(%[[ARG0:.*]]: !tt.ptr<bf16>) {
// CHECK:   %[[PID:.*]] = tt.get_program_id x : i32
// CHECK:   %[[PID_IDX:.*]] = arith.index_cast %[[PID]] : i32 to index
// CHECK:   %[[TILE:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME: as memref<1024x32x1x1xbf16, #xtile.layout<[3, 2, 0, 1]>>
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

!memref_type = memref<32xf64, #nvvm.memory_space<global>>
// CHECK:func.func @insert_extract_with_opaque_arg(
// CHECK-SAME: %[[ARG0:.*]]: !tt.ptr<f64>, %[[ARG1:.*]]: !tt.ptr<f64>, %[[ARG2:.*]]: i32) {
xtile.entry_func @insert_extract_with_opaque_arg(%input: !memref_type,
                                                 %output: !memref_type,
                                                 %opaque_arg: i32,
                                                 %tile_id: index) attributes {
                                                   num_opaque_args = 1: i32} {
  // CHECK: %[[SCALAR_VALUE:.*]] = tt.load %[[ARG0]] : !tt.ptr<f64>
  %tile = xtile.extract %input[%tile_id][1][1] : !memref_type -> tensor<f64>
  // CHECK: tt.store %[[ARG1]], %[[SCALAR_VALUE]] : !tt.ptr<f64>
  xtile.insert %tile into %output[%tile_id][1][1] : tensor<f64> -> !memref_type
  xtile.return
}

// -----

// CHECK-LABEL: func.func @fold_transpose_into_ptr
// CHECK-SAME: (%[[ARG0:.*]]: memref<32x16xf64, #xtile.layout<[0, 1]>>)
func.func @fold_transpose_into_ptr(
    %arg0: memref<32x16xf64, #xtile.layout<[0, 1]>>) -> !tt.ptr<f64> {
  %transposed = memref.transpose %arg0 (d0, d1) -> (d1, d0)
    : memref<32x16xf64, #xtile.layout<[0, 1]>> to memref<16x32xf64>
  // CHECK: %[[PTR:.*]] = triton_xla.memref_to_ptr %[[ARG0]] from memref<32x16xf64, #xtile.layout<[0, 1]>> to <f64>
  %ptr = triton_xla.memref_to_ptr %transposed from memref<16x32xf64> to !tt.ptr<f64>
  // CHECK: return %[[PTR]] : !tt.ptr<f64>
  return %ptr : !tt.ptr<f64>
}

// -----

// CHECK-LABEL: @mask_lowers_to_stable_hlo(%arg0: tensor<32xf64>, %arg1: f64) -> tensor<32xf64>
func.func @mask_lowers_to_stable_hlo(%arg0: tensor<32xf64>, %arg1: f64) -> tensor<32xf64> {
  // CHECK: %[[BOUND:.*]] = arith.constant dense<10> : tensor<32xi32>
  // CHECK: %[[IDX:.*]] = stablehlo.iota dim = 0 : tensor<32xi32>
  // CHECK: %[[IDX_BROADCASTED:.*]] = stablehlo.broadcast_in_dim %[[IDX]],
  // CHECK-SAME: dims = [0] : (tensor<32xi32>) -> tensor<32xi32>
  // CHECK: %[[MASK:.*]] = arith.cmpi slt, %[[IDX_BROADCASTED]], %[[BOUND]]
  // CHECK-SAME: : tensor<32xi32>
  // CHECK: %[[INIT:.*]] = tensor.from_elements %arg1 : tensor<f64>
  // CHECK: %[[INIT_TENSOR:.*]] = stablehlo.broadcast_in_dim %[[INIT]],
  // CHECK-SAME: dims = [] : (tensor<f64>) -> tensor<32xf64>
  // CHECK: %[[RESULT:.*]] = arith.select %[[MASK]], %arg0, %[[INIT_TENSOR]]
  // CHECK-SAME: : tensor<32xi1>, tensor<32xf64>
  %paded = xtile.mask %arg0 bounds [10], %arg1 : tensor<32xf64>
  // CHECK: return %[[RESULT]] : tensor<32xf64>
  return %paded : tensor<32xf64>
}

