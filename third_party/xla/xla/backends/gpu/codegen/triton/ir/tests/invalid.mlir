// RUN: xla-opt --verify-diagnostics %s

tt.func @extract_0d(%arg0: !tt.ptr<bf16>) {
  // expected-error @+1 {{cannot extract a 0-d tensor}}
  %0 = triton_xla.extract %arg0 [][][]
    {shape = array<i64>, layout = array<i64>} : !tt.ptr<bf16> to tensor<bf16>
  tt.return
}

tt.func @insert_0d(%arg0: tensor<bf16>, %arg1: !tt.ptr<bf16>) {
  // expected-error @+1 {{cannot insert a 0-d tensor}}
  triton_xla.insert %arg0 into %arg1 [][][]
    {shape = array<i64>, layout = array<i64>} : tensor<bf16> into !tt.ptr<bf16>
  tt.return
}

tt.func @extract_wrong_shape(%arg0: !tt.ptr<bf16>) {
  // expected-error @+1 {{shape attribute has a wrong size}}
  %1 = triton_xla.extract %arg0 [0][8][1]
    {shape = array<i64>, layout = array<i64:0>} : !tt.ptr<bf16> to tensor<8xbf16>
  tt.return
}

tt.func @extract_wrong_layout(%arg0: !tt.ptr<bf16>) {
  // expected-error @+1 {{layout attribute has a wrong size}}
  %0 = triton_xla.extract %arg0 [0][8][1]
    {shape = array<i64:8>, layout = array<i64>} : !tt.ptr<bf16> to tensor<8xbf16>
  tt.return
}

tt.func @insert_wrong_shape(%arg0: tensor<8xbf16>, %arg1: !tt.ptr<bf16>) {
  // expected-error @+1 {{shape attribute has a wrong size}}
  triton_xla.insert %arg0 into %arg1 [0][8][1]
    {shape = array<i64>, layout = array<i64:0>} : tensor<8xbf16> into !tt.ptr<bf16>
  tt.return
}

tt.func @insert_wrong_layout(%arg0: tensor<8xbf16>, %arg1: !tt.ptr<bf16>) {
  // expected-error @+1 {{layout attribute has a wrong size}}
  triton_xla.insert %arg0 into %arg1 [0][8][1]
    {shape = array<i64:8>, layout = array<i64>} : tensor<8xbf16> into !tt.ptr<bf16>
  tt.return
}

