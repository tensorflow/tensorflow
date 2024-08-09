// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text -with-layouts -print-layouts -verify-diagnostics --via-builder=false %s | FileCheck %s

// Note: exporting to HLO with layout and tiles is NOT supported with --via-builder=true.

module @entry attributes {
  mhlo.cross_program_prefetches = [],
  mhlo.is_dynamic = false,
  mhlo.use_auto_spmd_partitioning = false,
  mhlo.xla_entry_computation_parameter_layouts = [
    dense<[0, 1, 2]> : tensor<3xindex>,
    dense<[1, 2, 0]> : tensor<3xindex>,
    [
      dense<[1, 0]> : tensor<2xindex>,
      dense<[0, 1]> : tensor<2xindex>
    ],
    dense<> : tensor<0xindex>
  ],
  mhlo.xla_entry_computation_parameter_tiles = [[], [], [[], []], [dense<128> : tensor<1xindex>]],
  mhlo.xla_entry_computation_result_layout = dense<[2, 0, 1]> : tensor<3xindex>,
  mhlo.xla_entry_computation_result_tiles = []
} {
  func.func @main(
    %arg0: tensor<2x3x4xf32>,
    %arg1: tensor<2x3x4xf32>,
    %arg2: tuple<tensor<1x2xf32>, tensor<1x2xf32>>,
    %arg3: tensor<i32>
  ) -> tensor<2x3x4xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={
// CHECK-SAME:   (f32[2,3,4]{0,1,2},
// CHECK-SAME:    f32[2,3,4]{1,2,0},
// CHECK-SAME:   (f32[1,2]{1,0}, f32[1,2]{0,1}),
// CHECK-SAME:   s32[]{:T(128)})->f32[2,3,4]{2,0,1}
// CHECK-SAME: }

// CHECK: ENTRY %main.6 (Arg_0.1: f32[2,3,4], Arg_1.2: f32[2,3,4], Arg_2.3: (f32[1,2], f32[1,2]), Arg_3.4: s32[]) -> f32[2,3,4] {
// CHECK:   %Arg_2.3 = (f32[1,2]{1,0}, f32[1,2]{0,1}) parameter(2)
// CHECK:   %Arg_3.4 = s32[]{:T(128)} parameter(3)
// CHECK:   %Arg_0.1 = f32[2,3,4]{0,1,2} parameter(0)
// CHECK:   %Arg_1.2 = f32[2,3,4]{1,2,0} parameter(1)
// CHECK:   ROOT %add.5 = f32[2,3,4]{2,0,1} add(f32[2,3,4]{0,1,2} %Arg_0.1, f32[2,3,4]{1,2,0} %Arg_1.2)
// CHECK: }

// -----

module @entry attributes {
  mhlo.cross_program_prefetches = [],
  mhlo.is_dynamic = false,
  mhlo.use_auto_spmd_partitioning = false,
  mhlo.xla_entry_computation_parameter_layouts = [
    dense<[0, 1, 2]> : tensor<3xindex>,
    dense<[1, 2, 0]> : tensor<3xindex>,
    [
      dense<[1, 0]> : tensor<2xindex>,
      dense<[0, 1]> : tensor<2xindex>
    ],
    dense<> : tensor<0xindex>
  ],
  mhlo.xla_entry_computation_parameter_tiles = [
    [dense<[2, 3]> : tensor<2xindex>],
    [dense<[3, 4]> : tensor<2xindex>],
    [
      [dense<[1, 2]> : tensor<2xindex>],
      [dense<[2, 1]> : tensor<2xindex>]
    ],
    [dense<128> : tensor<1xindex>]
  ],
  mhlo.xla_entry_computation_result_layout = dense<[2, 0, 1]> : tensor<3xindex>,
  mhlo.xla_entry_computation_result_tiles = [dense<[2, 3]> : tensor<2xindex>]
} {
  func.func @main(
    %arg0: tensor<2x3x4xf32>,
    %arg1: tensor<2x3x4xf32>,
    %arg2: tuple<tensor<1x2xf32>, tensor<1x2xf32>>,
    %arg3: tensor<i32>
  ) -> tensor<2x3x4xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={
// CHECK-SAME:   (f32[2,3,4]{0,1,2:T(2,3)},
// CHECK-SAME:    f32[2,3,4]{1,2,0:T(3,4)},
// CHECK-SAME:   (f32[1,2]{1,0:T(1,2)}, f32[1,2]{0,1:T(2,1)}),
// CHECK-SAME:   s32[]{:T(128)})->f32[2,3,4]{2,0,1:T(2,3)}
// CHECK-SAME: }

// CHECK: ENTRY %main.6 (Arg_0.1: f32[2,3,4], Arg_1.2: f32[2,3,4], Arg_2.3: (f32[1,2], f32[1,2]), Arg_3.4: s32[]) -> f32[2,3,4] {
// CHECK:   %Arg_2.3 = (f32[1,2]{1,0:T(1,2)}, f32[1,2]{0,1:T(2,1)}) parameter(2)
// CHECK:   %Arg_3.4 = s32[]{:T(128)} parameter(3)
// CHECK:   %Arg_0.1 = f32[2,3,4]{0,1,2:T(2,3)} parameter(0)
// CHECK:   %Arg_1.2 = f32[2,3,4]{1,2,0:T(3,4)} parameter(1)
// CHECK:   ROOT %add.5 = f32[2,3,4]{2,0,1:T(2,3)} add(f32[2,3,4]{0,1,2:T(2,3)} %Arg_0.1, f32[2,3,4]{1,2,0:T(3,4)} %Arg_1.2)
// CHECK: }

// -----

// expected-error@+1{{Multi-level nested parameter layout is not supported}}
module @entry attributes {
  mhlo.xla_entry_computation_parameter_layouts = [
    [
      [
        dense<[1, 0]> : tensor<2xindex>,
        dense<[0, 1]> : tensor<2xindex>
      ]
    ]
  ]
} {
  func.func @main(
    %arg0: tuple<tensor<1x2xf32>, tensor<1x2xf32>>
  ) -> tuple<tensor<1x2xf32>, tensor<1x2xf32>> {
    return %arg0 : tuple<tensor<1x2xf32>, tensor<1x2xf32>>
  }
}

// -----

// expected-error@+1{{Multi-level nested parameter tile is not supported}}
module @entry attributes {
  mhlo.xla_entry_computation_parameter_tiles = [
    [
      [
        [dense<[1, 2]> : tensor<2xindex>],
        [dense<[2, 1]> : tensor<2xindex>]
      ]
    ]
  ]
} {
  func.func @main(
    %arg0: tuple<tensor<1x2xf32>, tensor<1x2xf32>>
  ) -> tuple<tensor<1x2xf32>, tensor<1x2xf32>> {
    return %arg0 : tuple<tensor<1x2xf32>, tensor<1x2xf32>>
  }
}

// -----

// expected-error@+1{{Multi-result layout is not supported}}
module @entry attributes {
  mhlo.xla_entry_computation_result_layout = [
    dense<[0, 1]> : tensor<2xindex>,
    dense<[1, 0]> : tensor<2xindex>
  ]
} {
  func.func @main(
    %arg0: tuple<tensor<1x2xf32>, tensor<1x2xf32>>
  ) -> tuple<tensor<1x2xf32>, tensor<1x2xf32>> {
    return %arg0 : tuple<tensor<1x2xf32>, tensor<1x2xf32>>
  }
}

// -----

// expected-error@+1{{layout has invalid tiles}}
module @entry attributes {
  mhlo.xla_entry_computation_result_tiles = [
    dense<[0, 1]> : tensor<2xindex>,
    dense<[1, 0]> : tensor<2xindex>
  ]
} {
  func.func @main(
    %arg0: tensor<1x2xf32>
  ) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }
}
