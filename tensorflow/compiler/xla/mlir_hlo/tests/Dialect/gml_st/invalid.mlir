// RUN: mlir-hlo-opt %s -split-input-file -verify-diagnostics


func.func @materialize_rank_mismatch(%tensor: tensor<?x?xf32>) {
  // expected-error @+1 {{expected result type = 'tensor<4xf32>' to match the inferred type = 'tensor<4x1xf32>'}}
  %0 = gml_st.materialize %tensor[0, 0][4, 1][1, 1]
     : tensor<?x?xf32> to tensor<4xf32>
}

// -----

func.func @materialize_inferred_type_mismatch(%tensor: tensor<?x?xf32>,
                                              %dim: index) {
  // expected-error @+1 {{expected result type = 'tensor<4x?xf32>' to match the inferred type = 'tensor<?x4xf32>}}
  %0 = gml_st.materialize %tensor[0, 0][%dim, 4][1, 1]
     : tensor<?x?xf32> to tensor<4x?xf32>
}

// -----

func.func @materialize_scalar_with_dynamic_tile(
    %tensor: tensor<?x?xf32>, %dim: index) {
  // expected-error @+1 {{expected tile type -9223372036854775808, 2 to have a single element shape}}
  %0 = gml_st.materialize %tensor[0, 0][%dim, 2][1, 1]
     : tensor<?x?xf32> to f32
}

// -----

func.func @materialize_scalar_with_nonsingle_element_tile(
    %tensor: tensor<?x?xf32>) {
  // expected-error @+1 {{expected tile type 1, 2 to have a single element shape}}
  %0 = gml_st.materialize %tensor[0, 0][1, 2][1, 2]
     : tensor<?x?xf32> to f32
}

// -----

func.func @materialize_scalar_element_type_mismatch(%tensor: tensor<?x?xf32>) {
  // expected-error @+1 {{expected the result type 'i32' to match source element type 'f32'}}
  %0 = gml_st.materialize %tensor[0, 0][1, 1][1, 1]
     : tensor<?x?xf32> to i32
}

// -----

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

func.func @for_loop_wrong_yield_target(
    %arg: tensor<8xf32>, %output: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %sum = gml_st.for (%i) = (%c0) to (%c8) step (%c4)
      outs(%out_ = %output : tensor<f32>) {
    %arg_sub = gml_st.materialize %arg[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %out_sub = gml_st.materialize %out_[][][]
      : tensor<f32> to tensor<f32>

    %result_sub = linalg.dot
        ins(%arg_sub, %arg_sub : tensor<4xf32>, tensor<4xf32>)
        outs(%out_sub : tensor<f32>) -> tensor<f32>

    %identity = gml_st.tile[][][] : !gml_st.tile<>
    // expected-error@+1 {{'gml_st.set_yield' op expected output block argument 0 to match set_yield destination}}
    gml_st.set_yield %result_sub into %output[%identity]
      : tensor<f32> into tensor<f32>[!gml_st.tile<>]
  } : tensor<f32>
  func.return %sum : tensor<f32>
}

// -----

func.func @yield_with_accumulator_mismatched_type(
    %arg: tensor<8xf32>, %output: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c4) {
    %arg_sub = gml_st.materialize %arg[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %out_sub = gml_st.materialize %output[][][]
      : tensor<f32> to tensor<f32>

    %result_sub = linalg.dot
       ins(%arg_sub, %arg_sub : tensor<4xf32>, tensor<4xf32>)
       outs(%out_sub : tensor<f32>) -> tensor<f32>

    %identity = gml_st.tile[][][] : !gml_st.tile<>
    // expected-error@+1 {{'gml_st.set_yield' op expected accumulator region to have 2 arguments of type 'tensor<f32>'}}
    gml_st.set_yield %result_sub into %output[%identity]
      acc (%in, %out: memref<f32>) {
        gml_st.yield %in : memref<f32>
      }: tensor<f32> into tensor<f32>[!gml_st.tile<>]
  } : tensor<f32>
  func.return %sum : tensor<f32>
}

// -----

func.func @for_loop_wrong_yield_operands(
    %arg: tensor<8xf32>, %output: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %sum = gml_st.for (%i) = (%c0) to (%c8) step (%c4)
      outs(%out_ = %output : tensor<f32>) {
    %arg_sub = gml_st.materialize %arg[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %out_sub = gml_st.materialize %out_[][][]
      : tensor<f32> to tensor<f32>

    %result_sub = linalg.dot
        ins(%arg_sub, %arg_sub : tensor<4xf32>, tensor<4xf32>)
        outs(%out_sub : tensor<f32>) -> tensor<f32>

    // expected-error@+1 {{'gml_st.set_yield' op expected to have at least 1 destination operand (currently 0)}}
    gml_st.set_yield
  } : tensor<f32>
  func.return %sum : tensor<f32>
}
