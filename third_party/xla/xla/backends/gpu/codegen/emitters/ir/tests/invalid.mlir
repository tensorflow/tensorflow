// RUN: emitters_opt  %s -split-input-file -verify-diagnostics

#map = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d1*32+d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 8], s0 in [0, 1], s1 in [0, 1]">
#map1 = #xla.indexing_map<"(d0, d1)[s0] -> (d0 mod 16 + s0, d1), domain: d0 in [0, 32], d1 in [0, 2], s0 in [0, 1]">

func.func @insert(%input: !xla_gpu.indexed_vector<32x64xf32, #map>,
    %i: index, %j: index, %output: tensor<32x64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{insert_op map must not have any symbols}}
  %0 = xla_gpu.insert %input(%i, %j) into %output at #map1
    : !xla_gpu.indexed_vector<32x64xf32, #map> -> tensor<32x64xf32>
  func.return %0 : tensor<32x64xf32>
}

// -----

#map = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d1*32+d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 8], s0 in [0, 1], s1 in [0, 1]">
#map1 = #xla.indexing_map<"(d0, d1, d2) -> (d0 mod 16, d1, d2), domain: d0 in [0, 32], d1 in [0, 2], d2 in [0, 5]">

func.func @insert(%input: !xla_gpu.indexed_vector<32x64xf32, #map>,
    %i: index, %j: index, %output: tensor<32x64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{source map result count must equal insert_op's map's dimension count}}
  %0 = xla_gpu.insert %input(%i, %j) into %output at #map1
    : !xla_gpu.indexed_vector<32x64xf32, #map> -> tensor<32x64xf32>
  func.return %0 : tensor<32x64xf32>
}

// -----

func.func @reduce_missing_combiner(%in0: tensor<16x8x4xf32>, %init0: f32,
    %in1: tensor<16x8x4xi32>, %init1: i32) -> (tensor<8xf32>, tensor<8xi32>) {
  // expected-error @+1 {{combiner `@add` not found}}
  %sum:2 = xla_gpu.reduce (%in0, %in1) inits(%init0, %init1) dimensions=[0, 2]
    combiner=@add {xla.range = [0 : index, 42 : index]}
    : tensor<16x8x4xf32>, tensor<16x8x4xi32> to tensor<8xf32>, tensor<8xi32>
  func.return %sum#0, %sum#1 : tensor<8xf32>, tensor<8xi32>
}

// -----

func.func @add(%a_acc: f32, %b_acc: f32, %a: f32, %b: f32)
    -> (f32, f32) {
  %0 = arith.addf %a_acc, %a : f32
  %1 = arith.addf %b_acc, %b : f32
  func.return %0, %1 : f32, f32
}
func.func @reduce_wrong_combiner_type(%in0: tensor<16x8x4xf32>, %init0: f32,
    %in1: tensor<16x8x4xi32>, %init1: i32) -> (tensor<8xf32>, tensor<8xi32>) {
  // expected-error @+1 {{combiner `@add expected to have type '(f32, i32, f32, i32) -> (f32, i32)' but got '(f32, f32, f32, f32) -> (f32, f32)'}}
  %sum:2 = xla_gpu.reduce (%in0, %in1) inits(%init0, %init1) dimensions=[0, 2]
    combiner=@add {xla.range = [0 : index, 42 : index]}
    : tensor<16x8x4xf32>, tensor<16x8x4xi32> to tensor<8xf32>, tensor<8xi32>
  func.return %sum#0, %sum#1 : tensor<8xf32>, tensor<8xi32>
}

// -----

func.func @reduce_init_type_mismatch(%in: tensor<16x8x4xf32>, %init: i32)
    -> (tensor<16x4xf32>) {
  // expected-error @+1 {{init type 'i32' does not match inferred type 'f32'}}
  %sum = "xla_gpu.reduce"(%in, %init)
    <{combiner = @add, dimensions = array<i64: 1>}>
    : (tensor<16x8x4xf32>, i32) -> tensor<16x4xf32>
  func.return %sum : tensor<16x4xf32>
}
