// RUN: xla-opt --split-input-file --verify-diagnostics %s

tt.func @sparse_dot(%lhs: tensor<128x32xbf16>, %rhs: tensor<64x128xbf16>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<128x4xi16> * tensor<64x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----
tt.func @sparse_dot_invalid_lhs_type(%lhs: tensor<128x32xf32>, %rhs: tensor<64x128xbf16>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  // expected-error @+1 {{element type of operand A is not supported}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xf32> meta tensor<128x4xi16> * tensor<64x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----
tt.func @sparse_dot_invalid_lhs_shape(%lhs: tensor<1x128x32xbf16>, %rhs: tensor<64x128xbf16>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  // expected-error @+1 {{shape of operand A is incorrect}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<1x128x32xbf16> meta tensor<128x4xi16> * tensor<64x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----
tt.func @sparse_dot_invalid_rhs_type(%lhs: tensor<128x32xbf16>, %rhs: tensor<64x128xf32>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  // expected-error @+1 {{element type of operand B is not supported}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<128x4xi16> * tensor<64x128xf32> -> tensor<128x128xf32>
  tt.return
}

// -----
tt.func @sparse_dot_invalid_rhs_shape(%lhs: tensor<128x32xbf16>, %rhs: tensor<1x64x128xbf16>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  // expected-error @+1 {{shape of operand B is incorrect}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<128x4xi16> * tensor<1x64x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----
tt.func @sparse_dot_invalid_acc_type(%lhs: tensor<128x32xbf16>, %rhs: tensor<64x128xbf16>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xbf16>
  // expected-error @+1 {{element type of operand C is not supported}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<128x4xi16> * tensor<64x128xbf16> -> tensor<128x128xbf16>
  tt.return
}

// -----
tt.func @sparse_dot_invalid_acc_shape(%lhs: tensor<128x32xbf16>, %rhs: tensor<64x128xbf16>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<16384xf32>
  // expected-error @+1 {{shape of operand C is incorrect}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<128x4xi16> * tensor<64x128xbf16> -> tensor<16384xf32>
  tt.return
}

// -----
tt.func @sparse_dot_mismatch_lhs_acc(%lhs: tensor<128x32xbf16>, %rhs: tensor<64x128xbf16>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<64x128xf32>
  // expected-error @+1 {{operand shape dimensions are incorrect}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<128x4xi16> * tensor<64x128xbf16> -> tensor<64x128xf32>
  tt.return
}

// -----
tt.func @sparse_dot_mismatch_rhs_acc(%lhs: tensor<128x32xbf16>, %rhs: tensor<64x128xbf16>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x64xf32>
  // expected-error @+1 {{operand shape dimensions are incorrect}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<128x4xi16> * tensor<64x128xbf16> -> tensor<128x64xf32>
  tt.return
}

// -----
tt.func @sparse_dot_mismatch_lhs_rhs(%lhs: tensor<128x32xbf16>, %rhs: tensor<32x128xbf16>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  // expected-error @+1 {{operand shape dimensions are incorrect}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<128x4xi16> * tensor<32x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----
tt.func @sparse_dot_mismatch_input_types(%lhs: tensor<128x32xf16>, %rhs: tensor<64x128xbf16>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  // expected-error @+1 {{operand element types do not match}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xf16> meta tensor<128x4xi16> * tensor<64x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----
tt.func @sparse_dot_invalid_meta_type(%lhs: tensor<128x32xbf16>, %rhs: tensor<64x128xbf16>, %meta: tensor<128x4xi8>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  // expected-error @+1 {{sparse metadata tensor is invalid}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<128x4xi8> * tensor<64x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----
tt.func @sparse_dot_invalid_meta_shape(%lhs: tensor<128x32xbf16>, %rhs: tensor<64x128xbf16>, %meta: tensor<512xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  // expected-error @+1 {{sparse metadata tensor is invalid}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<512xi16> * tensor<64x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----
tt.func @sparse_dot_mismatch_meta_noncontracting(%lhs: tensor<128x32xbf16>, %rhs: tensor<64x128xbf16>, %meta: tensor<64x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  // expected-error @+1 {{sparse metadata shape dimensions are incorrect}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<64x4xi16> * tensor<64x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----
tt.func @sparse_dot_mismatch_meta_contracting(%lhs: tensor<128x32xbf16>, %rhs: tensor<64x128xbf16>, %meta: tensor<128x8xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  // expected-error @+1 {{sparse metadata shape dimensions are incorrect}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16> meta tensor<128x8xi16> * tensor<64x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----
#mma0 = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#enc0 = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
tt.func @sparse_dot_encoding_operand_mismatch(%lhs: tensor<128x32xbf16, #enc0>, %rhs: tensor<64x128xbf16>, %meta: tensor<128x4xi16>) {
  %acc = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  // expected-error @+1 {{mismatching encoding between A and B operands}}
  %res = triton_xla.sparse_dot %lhs, %rhs, %acc, %meta : tensor<128x32xbf16, #enc0> meta tensor<128x4xi16> * tensor<64x128xbf16> -> tensor<128x128xf32>
  tt.return
}
