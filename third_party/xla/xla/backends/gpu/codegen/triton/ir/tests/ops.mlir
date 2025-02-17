// RUN: xla-opt %s --split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: xla-opt %s --split-input-file | xla-opt --split-input-file | FileCheck %s
// Verify the generic form can be parsed.
// RUN: xla-opt %s --split-input-file --mlir-print-op-generic | xla-opt --split-input-file | FileCheck %s

// CHECK-LABEL: xla_triton_tile
tt.func @xla_triton_tile(%arg0: tensor<512x128xbf16>)
    -> !triton_xla.tiled_tensor<16x64|512x128xbf16> {
  // CHECK: triton_xla.tile
  %tiled_tensor = triton_xla.tile %arg0 [0, 0] [16, 64] [128, 1]
    : !triton_xla.tiled_tensor<16x64|512x128xbf16>
  tt.return %tiled_tensor : !triton_xla.tiled_tensor<16x64|512x128xbf16>
}

// -----

// CHECK-LABEL: xla_triton_extract
tt.func @xla_triton_extract(%arg0: !triton_xla.tiled_tensor<16x64|512x128xbf16>)
    -> tensor<16x64xbf16> {
  %cst = arith.constant 0 : index
  %extracted_tensor = triton_xla.extract %arg0 [%cst, %cst]
    : tensor<512x128xbf16> to tensor<16x64xbf16>
  tt.return %extracted_tensor : tensor<16x64xbf16>
}
// CHECK: triton_xla.extract

// -----

// CHECK-LABEL: xla_triton_insert
tt.func @xla_triton_insert(%src: tensor<16x64xbf16>,
    %dst: !triton_xla.tiled_tensor<16x64|512x128xbf16>) -> tensor<512x128xbf16> {
  %cst = arith.constant 0 : index
  %updated_tensor = triton_xla.insert %src into %dst [%cst, %cst]
  : tensor<16x64xbf16> into tensor<512x128xbf16>
  tt.return %updated_tensor : tensor<512x128xbf16>
}
// CHECK: triton_xla.insert

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2],
  CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1],
  instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma, kWidth=2}>
#dot_meta_enc = #triton_xla.sparse_dot_meta<{parent=#mma}>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: sparse_xla_triton_op
  tt.func @sparse_xla_triton_op(%A_dot: tensor<32x32xf16, #dot_operand_a>,
   %B_dot: tensor<64x32xf16, #dot_operand_b>,
   %meta_reg: tensor<32x4xi16, #dot_meta_enc>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // CHECK-LABEL: triton_xla.sparse_dot
    %D = triton_xla.sparse_dot %A_dot, %B_dot, %acc, %meta_reg :
      tensor<32x32xf16, #dot_operand_a> meta tensor<32x4xi16,
      #dot_meta_enc> * tensor<64x32xf16, #dot_operand_b>
        -> tensor<32x32xf32, #mma>
    tt.return
  }
}
