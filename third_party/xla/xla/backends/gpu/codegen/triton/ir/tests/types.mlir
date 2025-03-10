// RUN: xla-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: @xla_tiled_tensor_type
tt.func private @xla_tiled_tensor_type(
  %arg0: !triton_xla.tiled_tensor<16x64|320x512xbf16>)

