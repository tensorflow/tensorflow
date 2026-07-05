// RUN: xla-opt %s -split-input-file -xla-gpu-insert-pdl \
// RUN:   | FileCheck %s --check-prefix=INSERT
// RUN: xla-opt %s -split-input-file -xla-gpu-insert-pdl \
// RUN:   -triton-xla-extract-insert-to-triton \
// RUN:   -xla-lower-pdl-wait \
// RUN:   | FileCheck %s --check-prefix=LOWER
// RUN: xla-opt %s -split-input-file -xla-gpu-insert-pdl \
// RUN:   -triton-xla-extract-insert-to-triton="allow_tma=1 num_stages=3" \
// RUN:   -xla-lower-pdl-wait \
// RUN:   | FileCheck %s --check-prefix=LOWER-TMA

func.func @basic(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<128xbf16, #xtile.layout<[0]>>
      [0] [16] [1] : tensor<16xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<256xbf16, #xtile.layout<[0]>>
      [0] [16] [1] : tensor<16xbf16>
  func.return
}

// INSERT-LABEL: func.func @basic(
// INSERT-NEXT:   xla_gpu.pdl_wait
// INSERT-NEXT:   triton_xla.extract
// INSERT-NEXT:   triton_xla.insert
// INSERT-NEXT:   return

// LOWER-LABEL:   tt.func @basic(
// LOWER:         nvvm.griddepcontrol wait
// LOWER-NOT:     nvvm.griddepcontrol wait
// LOWER:         tt.load
// LOWER:         tt.store
// LOWER:         tt.return

// LOWER-TMA-LABEL:   tt.func @basic(
// LOWER-TMA-SAME:    %arg0: !tt.tensordesc<16xbf16>
// LOWER-TMA-SAME:    %arg1: !tt.tensordesc<16xbf16>
// LOWER-TMA:         nvvm.griddepcontrol wait
// LOWER-TMA-NOT:     nvvm.griddepcontrol wait
// LOWER-TMA:         %[[LOAD:.*]] = tt.descriptor_load %arg0
// LOWER-TMA:         tt.descriptor_store %arg1[{{.*}}], %[[LOAD]]
// LOWER-TMA-NOT:     tt.load
// LOWER-TMA-NOT:     tt.store
// LOWER-TMA:         tt.return

// -----

module attributes {xla.pdl_launch} {
func.func @gemm_dot_launch_dependents(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>,
                                      %arg2: !tt.ptr<f32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
  %c0 = arith.constant 0 : index
  %lhs_3d = triton_xla.extract from %arg1
      as memref<8x128x1024xbf16, #xtile.layout<[2, 1, 0]>>
      [%c0, %c0, %c0] [1, 128, 64] [1, 1, 1]
      : tensor<1x128x64xbf16>
  %rhs_3d = triton_xla.extract from %arg0
      as memref<32x8x128xbf16, #xtile.layout<[2, 1, 0]>>
      [%c0, %c0, %c0] [32, 1, 128] [1, 1, 1]
      : tensor<32x1x128xbf16>
  %lhs_2d = tt.reshape %lhs_3d
      : tensor<1x128x64xbf16> -> tensor<128x64xbf16>
  %lhs = tt.trans %lhs_2d {order = array<i32: 1, 0>}
      : tensor<128x64xbf16> -> tensor<64x128xbf16>
  %rhs_2d = tt.reshape %rhs_3d
      : tensor<32x1x128xbf16> -> tensor<32x128xbf16>
  %rhs = tt.trans %rhs_2d {order = array<i32: 1, 0>}
      : tensor<32x128xbf16> -> tensor<128x32xbf16>
  %dot0 = tt.dot %lhs, %rhs, %cst, inputPrecision = tf32
      : tensor<64x128xbf16> * tensor<128x32xbf16> -> tensor<64x32xf32>
  %dot1 = tt.dot %lhs, %rhs, %dot0, inputPrecision = tf32
      : tensor<64x128xbf16> * tensor<128x32xbf16> -> tensor<64x32xf32>
  %out_3d = tt.reshape %dot1 : tensor<64x32xf32> -> tensor<1x64x32xf32>
  %out = tt.trans %out_3d {order = array<i32: 0, 2, 1>}
      : tensor<1x64x32xf32> -> tensor<1x32x64xf32>
  triton_xla.insert %out into %arg2
      as memref<8x32x1024xf32, #xtile.layout<[2, 1, 0]>>
      [%c0, %c0, %c0] [1, 32, 64] [1, 1, 1]
      : tensor<1x32x64xf32>
  func.return
}
}

// INSERT-LABEL: func.func @gemm_dot_launch_dependents(
// INSERT-NEXT:   xla_gpu.pdl_wait
// INSERT:        tt.dot
// INSERT-NEXT:   tt.dot
// INSERT-NEXT:   xla_gpu.pdl_launch_dependents
// INSERT-NEXT:   tt.reshape
// INSERT:        return

// LOWER-LABEL:   tt.func @gemm_dot_launch_dependents(
// LOWER:         nvvm.griddepcontrol wait
// LOWER:         tt.dot
// LOWER:         tt.dot
// LOWER-NEXT:    nvvm.griddepcontrol launch_dependents
// LOWER:         tt.return

// -----

module attributes {xla.pdl_launch} {
func.func @control_flow_dot_launch_dependents(%arg0: tensor<64x128xbf16>,
                                              %arg1: tensor<128x32xbf16>,
                                              %pred: i1) -> tensor<64x32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
  %out = scf.if %pred -> (tensor<64x32xf32>) {
    %dot = tt.dot %arg0, %arg1, %cst, inputPrecision = tf32
        : tensor<64x128xbf16> * tensor<128x32xbf16> -> tensor<64x32xf32>
    scf.yield %dot : tensor<64x32xf32>
  } else {
    scf.yield %cst : tensor<64x32xf32>
  }
  func.return %out : tensor<64x32xf32>
}
}

// INSERT-LABEL: func.func @control_flow_dot_launch_dependents(
// INSERT-NEXT:   xla_gpu.pdl_wait
// INSERT:        scf.if
// INSERT:        tt.dot
// INSERT-NEXT:   scf.yield
// INSERT:        } else {
// INSERT:        scf.yield
// INSERT:        }
// INSERT:        xla_gpu.pdl_launch_dependents
// INSERT-NEXT:   return
