// RUN: xla-opt --convert-unsupported-types %s | FileCheck %s

module {
  // CHECK:   xtile.entry_func @triton_fn(
  // CHECK-SAME:   %arg0: memref<64x512xf8E4M3FN, #xtile.layout<[1, 0]>>,
  // CHECK-SAME:   %arg1: memref<64x16xi8, #xtile.layout<[1, 0]>>,
  // CHECK-SAME:   %arg2: memref<512x64xf8E4M3FN, #xtile.layout<[1, 0]>>,
  // CHECK-SAME:   %arg3: memref<16x64xi8, #xtile.layout<[1, 0]>>,
  xtile.entry_func @triton_fn(
      %arg0: memref<64x512xf8E4M3FN, #xtile.layout<[1, 0]>>,
      %arg1: memref<64x16xf8E8M0FNU, #xtile.layout<[1, 0]>>,
      %arg2: memref<512x64xf8E4M3FN, #xtile.layout<[1, 0]>>,
      %arg3: memref<16x64xf8E8M0FNU, #xtile.layout<[1, 0]>>,
      %tile_id: index) {
    // CHECK-DAG: %[[C_0:.*]] = arith.constant 0 : index
    %c_0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %extracted_tile = xtile.extract %arg0[%c_0, %c_0] [16, 32] [1, 1] : memref<64x512xf8E4M3FN, #xtile.layout<[1, 0]>> -> tensor<16x32xf8E4M3FN>
    // CHECK: %[[arg_0:.*]] = xtile.extract %arg0[%[[C_0]], %[[C_0]]] [16, 32] [1, 1] : memref<64x512xf8E4M3FN, #xtile.layout<[1, 0]>> -> tensor<16x32xf8E4M3FN>
    %extracted_tile_0 = xtile.extract %arg1[%c_0, %c_0] [16, 1] [1, 1] : memref<64x16xf8E8M0FNU, #xtile.layout<[1, 0]>> -> tensor<16x1xf8E8M0FNU>
    // CHECK: %[[arg_1:.*]] = xtile.extract %arg1[%[[C_0]], %[[C_0]]] [16, 1] [1, 1] : memref<64x16xi8, #xtile.layout<[1, 0]>> -> tensor<16x1xi8>
    %extracted_tile_1 = xtile.extract %arg2[%c_0, %c_0] [32, 16] [1, 1] : memref<512x64xf8E4M3FN, #xtile.layout<[1, 0]>> -> tensor<32x16xf8E4M3FN>
    // CHECK: %[[arg_2:.*]] = xtile.extract %arg2[%[[C_0]], %[[C_0]]] [32, 16] [1, 1] : memref<512x64xf8E4M3FN, #xtile.layout<[1, 0]>> -> tensor<32x16xf8E4M3FN>
    %extracted_tile_2 = xtile.extract %arg3[%c_0, %c_0] [1, 16] [1, 1] : memref<16x64xf8E8M0FNU, #xtile.layout<[1, 0]>> -> tensor<1x16xf8E8M0FNU>
    // CHECK: %[[arg_3:.*]] = xtile.extract %arg3[%[[C_0]], %[[C_0]]] [1, 16] [1, 1] : memref<16x64xi8, #xtile.layout<[1, 0]>> -> tensor<1x16xi8>
    %16 = arith.bitcast %extracted_tile_0 : tensor<16x1xf8E8M0FNU> to tensor<16x1xi8>
    %17 = arith.bitcast %extracted_tile_2 : tensor<1x16xf8E8M0FNU> to tensor<1x16xi8>
    %18 = tt.trans %17 {order = array<i32: 1, 0>} : tensor<1x16xi8> -> tensor<16x1xi8>
    %19 = tt.dot_scaled %extracted_tile scale %16, %extracted_tile_1 scale %18, %cst lhs = e4m3 rhs = e4m3 {fastMath = true} : tensor<16x32xf8E4M3FN>, tensor<16x1xi8> * tensor<32x16xf8E4M3FN>, tensor<16x1xi8> -> tensor<16x16xf32>
    // arith.constant with f8E8M0FNU dense → i8
    %cst_scale = arith.constant dense<5.877470e-39> : tensor<16x1xf8E8M0FNU>
    // CHECK-DAG: %[[CST_SCALE:.*]] = arith.constant dense<0> : tensor<16x1xi8>
    // tensor.extract from 0-d f8E8M0FNU tensor → i8
    %cst_scalar = arith.constant dense<5.877470e-39> : tensor<f8E8M0FNU>
    // CHECK-DAG: %[[CST_SCALAR:.*]] = arith.constant dense<0> : tensor<i8>
    %extracted_scalar = tensor.extract %cst_scalar[] : tensor<f8E8M0FNU>
    // CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[CST_SCALAR]][] : tensor<i8>
    // arith.select with f8E8M0FNU operands → i8 (mask derived from
    // non-constant values to prevent constant folding)
    %mask = arith.cmpi sge, %16, %18 : tensor<16x1xi8>
    // CHECK: %[[MASK:.*]] = arith.cmpi sge, {{.*}} : tensor<16x1xi8>
    %selected = arith.select %mask, %extracted_tile_0, %cst_scale : tensor<16x1xi1>, tensor<16x1xf8E8M0FNU>
    // CHECK: %[[SELECTED:.*]] = arith.select %[[MASK]], %[[arg_1]], %[[CST_SCALE]] : tensor<16x1xi1>, tensor<16x1xi8>
    xtile.return
  }
}
