// RUN: mlir-hlo-opt %s -split-input-file -verify-diagnostics

func.func @tile_op_mismatch_sizes_and_static_sizes(%i: index) {
  // expected-error@+1 {{expected 0 dynamic size values}}
  %1 = "gml_st.tile"(%i) { static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1>, static_strides = array<i64: 1, 1>, operand_segment_sizes = array<i32: 0, 1, 0> } : (index) -> !gml_st.tile<?x?>
  func.return
}

// -----

func.func @tile_op_mismatch_offsets_and_static_offsets(%i: index) -> !gml_st.tile<8x8> {
  // expected-error@+1 {{expected 0 dynamic offset values}}
  %1 = "gml_st.tile"(%i) {static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 8, 8>, static_strides = array<i64: 1, 1>, operand_segment_sizes = array<i32: 1, 0, 0>} : (index) -> !gml_st.tile<8x8>
  func.return %1 : !gml_st.tile<8x8>
}

// -----

func.func @tile_op_mismatch_strides_and_static_strides(%i: index)  -> !gml_st.tile<8x8> {
  // expected-error@+1 {{expected 0 dynamic stride values}}
  %1 = "gml_st.tile"(%i) {static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 8, 8>, static_strides = array<i64: 1, 1>, operand_segment_sizes = array<i32: 0, 0, 1>} : (index) -> !gml_st.tile<8x8>
  func.return %1 : !gml_st.tile<8x8>
}

// -----

func.func @tile_op_negative_static_size(%i: index)  -> !gml_st.tile<?x?> {
  // expected-error@+1 {{'gml_st.tile' op expected size = -2 to be non-negative}}
  %1 = "gml_st.tile"(%i) {static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -2>, static_strides = array<i64: 1, 1>, operand_segment_sizes = array<i32: 0, 1, 0>} : (index) -> !gml_st.tile<?x?>
  func.return %1 : !gml_st.tile<?x?>
}

// -----

func.func @tile_op_negative_static_stride(%i: index)  -> !gml_st.tile<?x8> {
  // expected-error@+1 {{'gml_st.tile' op expected stride = -2 to be non-negative}}
  %1 = "gml_st.tile"(%i) {static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, 8>, static_strides = array<i64: 1, -2>, operand_segment_sizes = array<i32: 0, 1, 0>} : (index) -> !gml_st.tile<?x8>
  func.return %1 : !gml_st.tile<?x8>
}

// -----

func.func @tile_op_negative_static_offset(%i: index)  -> !gml_st.tile<?x8> {
  // expected-error@+1 {{'gml_st.tile' op expected offset = -2 to be non-negative}}
  %1 = "gml_st.tile"(%i) {static_offsets = array<i64: 0, -2>, static_sizes = array<i64: -9223372036854775808, 8>, static_strides = array<i64: 1, 1>, operand_segment_sizes = array<i32: 0, 1, 0>} : (index) -> !gml_st.tile<?x8>
  func.return %1 : !gml_st.tile<?x8>
}

// -----

func.func @yield_with_accumulator_mismatched_type(
    %arg: tensor<8xf32>, %output: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c4)
      outs (%out_ = %output: tensor<f32>) {
    %arg_sub = tensor.extract_slice %arg[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %out_sub = tensor.extract_slice %out_[][][]
      : tensor<f32> to tensor<f32>

    %result_sub = linalg.dot
       ins(%arg_sub, %arg_sub : tensor<4xf32>, tensor<4xf32>)
       outs(%out_sub : tensor<f32>) -> tensor<f32>

    %identity = gml_st.tile[][][] : !gml_st.tile<>
    // expected-error@+1 {{'gml_st.set_yield' op expected accumulator region to have 2 arguments of type 'tensor<f32>'}}
    gml_st.set_yield %result_sub into %out_[%identity]
      acc (%in, %out: memref<f32>) {
        gml_st.yield %in : memref<f32>
      }: tensor<f32> into tensor<f32>[!gml_st.tile<>]
  } : tensor<f32>
  func.return %sum : tensor<f32>
}

// -----

func.func @missing_output_tensors(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  // expected-error@+1 {{expected the number of output arguments to match the number of results}}
  %13 = gml_st.parallel (%arg4, %arg5) = (%c0, %c16) to (%c1, %c16)
        step (%c8, %c8) {
    %19 = gml_st.tile [%arg4, %arg5] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %11 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    gml_st.set_yield %11 into %0[%19] : tensor<8x8xf32>
          into tensor<8x8xf32>[!gml_st.tile<8x8>]
  } : tensor<8x8xf32>
  return %13 : tensor<8x8xf32>
}
