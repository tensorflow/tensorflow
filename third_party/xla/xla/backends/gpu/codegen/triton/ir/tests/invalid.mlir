// RUN: xla-opt --split-input-file --verify-diagnostics %s

// TODO(manany): Fix issue with this test.
tt.func @tile_mismatch_rank(%arg0: tensor<256x256xbf16>) {
  %cst_0_i32 = arith.constant 0 : i32
  %cst_0_i64 = arith.constant 0 : i64
  %cst_64_i64 = arith.constant 64 : i64
  // expected-error @+1 {{mismatch between tensor rank and one or more of offsets/sizes/strides}}
  %tiled_tensor = triton_xla.tile %arg0 [%cst_0_i32][%cst_64_i64][%cst_0_i64]
        : !triton_xla.tiled_tensor<16x64|256x256xbf16>
  tt.return
}

// -----

tt.func @extract_mismatch_rank(
        %arg0: !triton_xla.tiled_tensor<16x64|256x256xbf16>) {
  %cst = arith.constant 0 : i32
  // expected-error @+1 {{source tensor rank does not match number of offsets}}
  %extracted_tensor = triton_xla.extract %arg0 [%cst]
        : tensor<256x256xbf16> to tensor<16x64xbf16>
  tt.return
}

// -----

tt.func @insert_mismatch_rank(
        %arg0: tensor<16x64xbf16>,
        %arg1: !triton_xla.tiled_tensor<16x64|256x256xbf16>) {
  %cst = arith.constant 0 : i32
  // expected-error @+1 {{destination tensor rank does not match number of offsets}}
  %inserted_tensor = triton_xla.insert %arg0 into %arg1 [%cst,%cst,%cst]
        : tensor<16x64xbf16> into tensor<256x256xbf16>
  tt.return
}

// -----

"tt.func"() <{function_type = (tensor<bf16>) -> !triton_xla.tiled_tensor<|bf16>, sym_name = "xla_triton_tile"}> ({
^bb0(%arg0: tensor<bf16>):
  // expected-error @+1 {{cannot tile a 0-d tensor}}
  %0 = "triton_xla.tile"(%arg0) : (tensor<bf16>) -> !triton_xla.tiled_tensor<|bf16>
  "tt.return"(%0) : (!triton_xla.tiled_tensor<|bf16>) -> ()
}) : () -> ()

// -----

"tt.func"() <{function_type = (!triton_xla.tiled_tensor<|bf16>) -> tensor<bf16>, sym_name = "xla_triton_extract"}> ({
^bb0(%arg0: !triton_xla.tiled_tensor<|bf16>):
  %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  // expected-error @+1 {{cannot extract a 0-d tensor}}
  %1 = "triton_xla.extract"(%arg0, %0, %0) : (!triton_xla.tiled_tensor<|bf16>, i32, i32) -> tensor<bf16>
  "tt.return"(%1) : (tensor<bf16>) -> ()
}) : () -> ()

// -----

"tt.func"() <{function_type = (tensor<bf16>, !triton_xla.tiled_tensor<|bf16>) -> tensor<bf16>, sym_name = "xla_triton_insert"}> ({
^bb0(%arg0: tensor<bf16>, %arg1: !triton_xla.tiled_tensor<|bf16>):
  %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  // expected-error @+1 {{cannot insert a 0-d tensor}}
  %1 = "triton_xla.insert"(%arg0, %arg1, %0, %0) : (tensor<bf16>, !triton_xla.tiled_tensor<|bf16>, i32, i32) -> tensor<bf16>
  "tt.return"(%1) : (tensor<bf16>) -> ()
}) : () -> ()
