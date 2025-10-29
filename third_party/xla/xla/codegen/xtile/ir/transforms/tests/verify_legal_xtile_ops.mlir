// RUN: emitters_opt %s -xtile-verify-legal-ops -split-input-file -verify-diagnostics

xtile.entry_func @fails_illegal_op(%arg0: memref<2xf32>, %arg1: index) {
  %c_0 = arith.constant 0. : f32
  // expected-error @+1 {{vector.transfer_read: unsupported op}}
  %0 = vector.transfer_read %arg0[%arg1], %c_0 : memref<2xf32>, vector<2xf32>
  // expected-error @+1 {{vector.transfer_write: unsupported op}}
  vector.transfer_write %0, %arg0[%arg1] : vector<2xf32>, memref<2xf32>
  xtile.return
}

// -----
