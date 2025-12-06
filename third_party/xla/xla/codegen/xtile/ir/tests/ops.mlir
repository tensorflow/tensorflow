// RUN: emitters_opt %s --split-input-file --verify-roundtrip -verify-diagnostics

xtile.entry_func @happy_path(%input: memref<1024x4xf32>, %output: memref<128x1024xf32>, %tile_id: index) {
  %tile = xtile.extract %input[%tile_id, %tile_id][10, 1][1, 1] : memref<1024x4xf32> -> tensor<10xf32>
  xtile.insert %tile into %output[%tile_id, %tile_id][10, 1][1, 1] : tensor<10xf32> -> memref<128x1024xf32>
  xtile.return
}

// -----

xtile.entry_func @with_attributes(
  %input: memref<1024xf32> {xla.some_attr = 1},
  %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:10, tiles_per_workgroup:5>} {
  xtile.return
}

// -----

// expected-error@+1 {{entry function arguments should be of the form (arg: memref..., tile_id: index)}}
xtile.entry_func @tile_id_at_start(%tile_id: index, %input: memref<1024xf32>, %output: memref<1024xf32>) {
  xtile.return
}

// -----

// expected-error@+1 {{entry function arguments should be of the form (arg: memref..., tile_id: index)}}
xtile.entry_func @too_many_tile_ids(%input: memref<1024xf32>, %id0: index, %id1: index) {
  xtile.return
}

// -----

xtile.entry_func @correct_opaque_args(
  %input: memref<1024xf32>, %opaque0: index, %opaque1: index, %id1: index)
  attributes {num_opaque_args = 2 : i32}  {
  xtile.return
}

// -----

// expected-error@+1 {{entry function arguments should be of the form (arg: memref..., tile_id: index)}}
xtile.entry_func @wrong_opaque_args(
  %input: memref<1024xf32>, %opaque0: index, %opaque1: index, %id1: index)
  attributes {num_opaque_args = 1 : i32}  {
  xtile.return
}

// -----

func.func @incorrect_full_shape_extract(%arg: memref<1024xf32>) -> tensor<10xf32> {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{full tile shape size: 2 does not match rank of buffer: 1}}
  %tile = xtile.extract %arg[%offset][10, 1][1] : memref<1024xf32> -> tensor<10xf32>
  return %tile : tensor<10xf32>
}

// -----

func.func @incorrect_offset_count_extract(%arg: memref<1024xf32>) -> tensor<10xf32> {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{expected 1 offset operands, got 2}}
  %tile = xtile.extract %arg[%offset, %offset][10][1] : memref<1024xf32> -> tensor<10xf32>
  return %tile : tensor<10xf32>
}

// -----

func.func @incorrect_rank_reduction_extract(%arg: memref<16x1024xf32>) -> tensor<10xf32> {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{full tile shape: [16, 10] does not reduce to tile shape: [10]}}
  %tile = xtile.extract %arg[%offset, %offset][16, 10][1, 1] : memref<16x1024xf32> -> tensor<10xf32>
  return %tile : tensor<10xf32>
}

// -----

func.func @type_mismatch_extract(%arg: memref<1024xf32>) -> tensor<10xf64> {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{buffer element type: 'f32' does not match element type of tile: 'f64'}}
  %tile = xtile.extract %arg[%offset][10][1] : memref<1024xf32> -> tensor<10xf64>
  return %tile : tensor<10xf64>
}

// -----

func.func @incorrect_full_shape_insert(%src: tensor<24xf32>, %dst: memref<1024xf32>) {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{full tile shape size: 2 does not match rank of buffer: 1}}
  xtile.insert %src into %dst[%offset][24, 1][1] : tensor<24xf32> -> memref<1024xf32>
  return
}

// -----

func.func @incorrect_offset_count_insert(%src: tensor<24xf32>, %dst: memref<1024xf32>) {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{expected 1 offset operands, got 2}}
  xtile.insert %src into %dst[%offset, %offset][24][1] : tensor<24xf32> -> memref<1024xf32>
  return
}

// -----

func.func @incorrect_rank_reduction_insert(%src: tensor<24xf32>, %dst: memref<16x1024xf32>) {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{full tile shape: [16, 24] does not reduce to tile shape: [24]}}
  xtile.insert %src into %dst[%offset, %offset][16, 24][1, 1] : tensor<24xf32> -> memref<16x1024xf32>
  return
}

// -----

func.func @type_mismatch_insert(%src: tensor<24xf64>, %dst: memref<1024xf32>) {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{buffer element type: 'f32' does not match element type of tile: 'f64'}}
  xtile.insert %src into %dst[%offset][24][1] : tensor<24xf64> -> memref<1024xf32>
  return
}

// -----

func.func @dot_scaled(%lhs: tensor<128x128xf32>, %lhs_scale: tensor<128x4xi8>, %rhs: tensor<128x256xf32>, %rhs_scale: tensor<256x4xi8>, %acc: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = xtile.dot_scaled %lhs scale %lhs_scale, %rhs scale %rhs_scale {fastMath = true} : tensor<128x128xf32>, tensor<128x4xi8> * tensor<128x256xf32>, tensor<256x4xi8> -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}


// -----

func.func @legal_mask_op(%src: tensor<32xf64>, %mask: f64) -> tensor<32xf64> {
  %masked = xtile.mask %src bounds [10], %mask : tensor<32xf64>
  return %masked : tensor<32xf64>
}

// -----

func.func @illegal_mask_bound_rank_mismatch(
    %src: tensor<32xf64>, %mask: f64) -> tensor<32xf64> {
  // expected-error@+1 {{tensor rank: 1 does not match mask bounds rank: 2}}
  %masked = xtile.mask %src bounds [10, 1], %mask : tensor<32xf64>
  return %masked : tensor<32xf64>
}

// -----

func.func @illegal_mask_out_of_bounds(%src: tensor<32xf64>, %mask: f64) -> tensor<32xf64> {
  // expected-error@+1 {{mask bound not less than or equal to the tensor size}}
  %masked = xtile.mask %src bounds [33], %mask : tensor<32xf64>
  return %masked : tensor<32xf64>
}

// -----

// expected-error @+1 {{layout has 0 dimensions, but shape has 1}}
func.func @memref_layout_shape_size_mismatch(%arg0: memref<1024xf32, #xtile.layout<[]>>) {
  return
}

// -----

// expected-error @+1 {{layout is not a permutation}}
func.func @memref_layout_is_not_a_permutation(%arg0: memref<1024xf32, #xtile.layout<[1]>>) {
  return
}
