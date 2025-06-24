// RUN: xla-opt --split-input-file --verify-diagnostics %s

tt.func @extract_0d(%arg0: tensor<bf16>) {
  // expected-error @+1 {{cannot extract a 0-d tensor}}
  %extracted_tensor = triton_xla.extract %arg0 [][][]
    {layout = array<i64:1, 0>} : tensor<bf16> to tensor<bf16>
  tt.return
}

// -----

tt.func @insert_0d(%arg0: tensor<bf16>, %arg1: tensor<bf16>) {
  %cst = arith.constant 0 : index
  // expected-error @+1 {{cannot insert a 0-d tensor}}
  %inserted_tensor = triton_xla.insert %arg0 into %arg1 [][][]
    {layout = array<i64:1, 0>} : tensor<bf16> into tensor<bf16>
  tt.return
}

// -----

tt.func @extract_wrong_layout(%arg0: tensor<16xbf16>) {
  // expected-error @+1 {{layout attribute has a wrong size}}
  %extracted_tensor = triton_xla.extract %arg0 [0][8][1]
    {layout = array<i64:1, 0>} : tensor<16xbf16> to tensor<8xbf16>
  tt.return
}

// -----

tt.func @insert_wrong_layout(%arg0: tensor<8xbf16>, %arg1: tensor<16xbf16>) {
  %cst = arith.constant 0 : index
  // expected-error @+1 {{layout attribute has a wrong size}}
  %inserted_tensor = triton_xla.insert %arg0 into %arg1 [0][8][1]
    {layout = array<i64:1, 0>} : tensor<8xbf16> into tensor<16xbf16>
  tt.return
}

