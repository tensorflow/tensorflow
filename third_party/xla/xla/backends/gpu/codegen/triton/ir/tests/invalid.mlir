// RUN: xla-opt --split-input-file --verify-diagnostics %s

tt.func @extract_mismatch_rank(%arg0: tensor<256x256xbf16>) {
  %cst = arith.constant 0 : index
  // expected-error @+1 {{ranks of source/destination tensor and offsets/strides do not match}}
  %extracted_tensor = triton_xla.extract %arg0 [%cst][%cst]
      {layout = array<i64:1, 0>} : tensor<256x256xbf16> to tensor<16x64xbf16>
  tt.return
}

// -----

tt.func @insert_mismatch_rank(
        %arg0: tensor<16x64xbf16>,
        %arg1: tensor<256x256xbf16>) {
  %cst = arith.constant 0 : index
  // expected-error @+1 {{ranks of source/destination tensor and offsets/strides do not match}}
  %inserted_tensor = triton_xla.insert %arg0 into %arg1 [%cst,%cst,%cst][%cst,%cst,%cst]
      {layout = array<i64:1, 0>} : tensor<16x64xbf16> into tensor<256x256xbf16>
  tt.return
}

// -----

"tt.func"() <{function_type = (tensor<bf16>) -> tensor<bf16>, sym_name = "xla_triton_extract"}> ({
^bb0(%arg0: tensor<bf16>):
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  // expected-error @+1 {{cannot extract a 0-d tensor}}
  %1 = "triton_xla.extract"(%arg0, %0, %0)  {layout = array<i64:1, 0>}
      : (tensor<bf16>, index, index) -> tensor<bf16>
  "tt.return"(%1) : (tensor<bf16>) -> ()
}) : () -> ()

// -----

"tt.func"() <{function_type = (tensor<bf16>, tensor<bf16>) -> tensor<bf16>, sym_name = "xla_triton_insert"}> ({
^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  // expected-error @+1 {{cannot insert a 0-d tensor}}
  %1 = "triton_xla.insert"(%arg0, %arg1, %0, %0)  {layout = array<i64:1, 0>}
      : (tensor<bf16>, tensor<bf16>, index, index) -> tensor<bf16>
  "tt.return"(%1) : (tensor<bf16>) -> ()
}) : () -> ()
