// RUN: hlo-translate -hlo-to-mlir -split-input-file -verify-diagnostics %s | FileCheck %s

HloModule main, entry_computation_layout={(f32[16,50]{1,0}, s64[1,<=16]{1,0})->f32[<=16,50]{1,0}}

// CHECK-LABEL: main
// CHECK:      stablehlo.reshape {{.*}} (tensor<1x?xi64, #stablehlo.bounds<?, 16>>) -> tensor<?xi64, #stablehlo.bounds<16>>
// CHECK-NEXT: "stablehlo.gather"{{.*}} : (tensor<16x50xf32>, tensor<?xi64, #stablehlo.bounds<16>>) -> tensor<?x50xf32, #stablehlo.bounds<16, ?>>
ENTRY %main.5 (Arg_0.1: f32[16,50], Arg_1.2: s64[1,<=16]) -> f32[<=16,50] {
  %Arg_0.1 = f32[16,50] parameter(0)
  %Arg_1.2 = s64[1,<=16] parameter(1)
  %reshape.3 = s64[<=16] reshape(%Arg_1.2), metadata={source_file="/tmp/t.mlir" source_line=3}
  ROOT %gather.4 = f32[<=16,50] gather(%Arg_0.1, %reshape.3), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,50}, metadata={source_file="/tmp/t.mlir" source_line=4}
}
