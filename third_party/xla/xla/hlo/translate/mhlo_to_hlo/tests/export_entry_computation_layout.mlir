// RUN: xla-translate -mlir-hlo-to-hlo-text -with-layouts -print-layouts -split-input-file -verify-diagnostics --via-builder=false %s | FileCheck %s

// Note: exporting to HLO with layout and tiles is NOT supported with --via-builder=true.

module @entry attributes {
  mhlo.xla_entry_computation_parameter_layouts = [
    dense<[0, 1, 2]> : tensor<3xindex>,
    dense<[1, 2, 0]> : tensor<3xindex>,
    [
      dense<[1, 0]> : tensor<2xindex>,
      dense<[0, 1]> : tensor<2xindex>
    ],
    []
  ],
  mhlo.xla_entry_computation_parameter_tiles = [
    [dense<[2, 128]> : tensor<2xindex>],
    [dense<[2, 1]> : tensor<2xindex>],
    [
      [dense<[1, 2]> : tensor<2xindex>],
      [dense<[2, 1]> : tensor<2xindex>]
    ],
    []
  ],
  mhlo.xla_entry_computation_result_layout = [dense<[2, 0, 1]> : tensor<3xindex>],
  mhlo.xla_entry_computation_result_tiles = [[dense<[2, 128]> : tensor<2xindex>]]
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

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[2,3,4]{0,1,2:T(2,128)},
// CHECK-SAME:   f32[2,3,4]{1,2,0:T(2,1)},
// CHECK-SAME:   (f32[1,2]{1,0:T(1,2)}, f32[1,2]{0,1:T(2,1)}),
// CHECK-SAME:   s32[]
// CHECK-SAME: )->f32[2,3,4]{2,0,1:T(2,128)}}

// CHECK: ENTRY %main.6 (Arg_0.1: f32[2,3,4], Arg_1.2: f32[2,3,4], Arg_2.3: (f32[1,2], f32[1,2]), Arg_3.4: s32[]) -> f32[2,3,4] {
// CHECK:   %Arg_2.3 = (f32[1,2]{1,0}, f32[1,2]{1,0}) parameter(2)
// CHECK:   %Arg_3.4 = s32[] parameter(3)
// CHECK:   %Arg_0.1 = f32[2,3,4]{2,1,0} parameter(0)
// CHECK:   %Arg_1.2 = f32[2,3,4]{2,1,0} parameter(1)
// CHECK:   ROOT %add.5 = f32[2,3,4]{2,1,0} add(f32[2,3,4]{2,1,0} %Arg_0.1, f32[2,3,4]{2,1,0} %Arg_1.2)
// CHECK: }

// -----

///////////////////////////////////////////
// Test Parameter Layouts                //
///////////////////////////////////////////

// Test layout
module @entry attributes {
  mhlo.xla_entry_computation_parameter_layouts = [
    dense<[1, 0, 2]> : tensor<3xindex>
  ]
} {
  func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    return %arg0 : tensor<2x3x4xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[2,3,4]{1,0,2}
// CHECK-SAME: )->f32[2,3,4]{2,1,0}

// CHECK: ENTRY %main.2 (Arg_0.1: f32[2,3,4]) -> f32[2,3,4] {
// CHECK:   ROOT %Arg_0.1 = f32[2,3,4]{2,1,0} parameter(0)
// CHECK: }

// -----

// Test no layout
// This is equivalent to mhlo.xla_entry_computation_parameter_layouts = []
module @entry attributes {} {
  func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    return %arg0 : tensor<2x3x4xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[2,3,4]{2,1,0}
// CHECK-SAME: )->f32[2,3,4]{2,1,0}

// CHECK: ENTRY %main.2 (Arg_0.1: f32[2,3,4]) -> f32[2,3,4] {
// CHECK:   ROOT %Arg_0.1 = f32[2,3,4]{2,1,0} parameter(0)
// CHECK: }

// -----

// Test empty layout
module @entry attributes {
  mhlo.xla_entry_computation_parameter_layouts = [
    dense<> : tensor<0xindex>
  ]
} {
  func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    return %arg0 : tensor<2x3x4xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[2,3,4]
// CHECK-SAME: )->f32[2,3,4]{2,1,0}

// CHECK: ENTRY %main.2 (Arg_0.1: f32[2,3,4]) -> f32[2,3,4] {
// CHECK:   ROOT %Arg_0.1 = f32[2,3,4]{2,1,0} parameter(0)
// CHECK: }

// -----

// Test nested tuple layout
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
    %arg0: tuple<tuple<tensor<1x2xf32>, tensor<1x2xf32>>>
  ) -> tuple<tuple<tensor<1x2xf32>, tensor<1x2xf32>>> {
    return %arg0 : tuple<tuple<tensor<1x2xf32>, tensor<1x2xf32>>>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   (
// CHECK-SAME:     (
// CHECK-SAME:       f32[1,2]{1,0},
// CHECK-SAME:       f32[1,2]{0,1}
// CHECK-SAME:     )
// CHECK-SAME:   )
// CHECK-SAME: )->((f32[1,2]{1,0}, f32[1,2]{1,0}))}

// CHECK: ENTRY %main.2 (Arg_0.1: ((f32[1,2], f32[1,2]))) -> ((f32[1,2], f32[1,2])) {
// CHECK:   ROOT %Arg_0.1 = ((f32[1,2]{1,0}, f32[1,2]{1,0})) parameter(0)
// CHECK: }

// -----

// Test multi arg layout
module @entry attributes {
  mhlo.xla_entry_computation_parameter_layouts = [
    dense<[1, 0, 2]> : tensor<3xindex>,
    dense<[0, 1, 2]> : tensor<3xindex>
  ]
} {
  func.func @main(
    %arg0: tensor<2x3x4xf32>,
    %arg1: tensor<2x3x4xf32>
  ) -> tensor<2x3x4xf32> {
    return %arg0 : tensor<2x3x4xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[2,3,4]{1,0,2},
// CHECK-SAME:   f32[2,3,4]{0,1,2}
// CHECK-SAME: )->f32[2,3,4]{2,1,0}}

// CHECK: ENTRY %main.3 (Arg_0.1: f32[2,3,4], Arg_1.2: f32[2,3,4]) -> f32[2,3,4] {
// CHECK:   ROOT %Arg_0.1 = f32[2,3,4]{2,1,0} parameter(0)
// CHECK:   %Arg_1.2 = f32[2,3,4]{2,1,0} parameter(1)
// CHECK: }

// -----

///////////////////////////////////////////
// Test Paramter Tiles                   //
///////////////////////////////////////////

// Test tile
module @entry attributes {
  mhlo.xla_entry_computation_parameter_tiles = [
    [dense<[2, 128]> : tensor<2xindex>]
  ]
} {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[1,2]{1,0:T(2,128)}
// CHECK-SAME: )->f32[1,2]{1,0}}

// CHECK: ENTRY %main.2 (Arg_0.1: f32[1,2]) -> f32[1,2] {
// CHECK:   ROOT %Arg_0.1 = f32[1,2]{1,0} parameter(0)
// CHECK: }

// -----

// Test no tile
module @entry attributes {
  mhlo.xla_entry_computation_parameter_tiles = [
    []
  ]
} {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[1,2]{1,0}
// CHECK-SAME: )->f32[1,2]{1,0}}

// CHECK: ENTRY %main.2 (Arg_0.1: f32[1,2]) -> f32[1,2] {
// CHECK:   ROOT %Arg_0.1 = f32[1,2]{1,0} parameter(0)
// CHECK: }

// -----

// Test empty tile
module @entry attributes {
  mhlo.xla_entry_computation_parameter_tiles = [
    [dense<> : tensor<0xindex>]
  ]
} {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[1,2]{1,0}
// CHECK-SAME: )->f32[1,2]{1,0}}

// CHECK: ENTRY %main.2 (Arg_0.1: f32[1,2]) -> f32[1,2] {
// CHECK:   ROOT %Arg_0.1 = f32[1,2]{1,0} parameter(0)
// CHECK: }

// -----

// Test nested tuple tile
module @entry attributes {
  mhlo.xla_entry_computation_parameter_tiles = [
    [
      [
        [dense<[2, 128]> : tensor<2xindex>]
      ]
    ]
  ]
} {
  func.func @main(
    %arg0: tuple<tuple<tensor<1x2xf32>>>
  ) -> tuple<tuple<tensor<1x2xf32>>> {
    return %arg0 : tuple<tuple<tensor<1x2xf32>>>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   (
// CHECK-SAME:     (
// CHECK-SAME:       f32[1,2]{1,0:T(2,128)}
// CHECK-SAME:     )
// CHECK-SAME:   )
// CHECK-SAME: )->((f32[1,2]{1,0}))}

// CHECK: ENTRY %main.2 (Arg_0.1: ((f32[1,2]))) -> ((f32[1,2])) {
// CHECK:   ROOT %Arg_0.1 = ((f32[1,2]{1,0})) parameter(0)
// CHECK: }

// -----

// Test multi arg tile
module @entry attributes {
  mhlo.xla_entry_computation_parameter_tiles = [
    [dense<[2, 128]> : tensor<2xindex>],
    [dense<[4, 128]> : tensor<2xindex>]
  ]
} {
  func.func @main(
    %arg0: tensor<1x2xf32>,
    %arg1: tensor<1x2xf32>
  ) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[1,2]{1,0:T(2,128)},
// CHECK-SAME:   f32[1,2]{1,0:T(4,128)}
// CHECK-SAME: )->f32[1,2]{1,0}}

// CHECK: ENTRY %main.3 (Arg_0.1: f32[1,2], Arg_1.2: f32[1,2]) -> f32[1,2] {
// CHECK:   ROOT %Arg_0.1 = f32[1,2]{1,0} parameter(0)
// CHECK:   %Arg_1.2 = f32[1,2]{1,0} parameter(1)
// CHECK: }

// -----

// Test sub-tiles
module @entry attributes {
  mhlo.xla_entry_computation_parameter_tiles = [
    [dense<128> : tensor<index>, dense<2> : tensor<1xindex>]
  ]
} {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[1,2]{1,0:T(128)(2)}
// CHECK-SAME: )->f32[1,2]{1,0}}

// CHECK: ENTRY %main.2 (Arg_0.1: f32[1,2]) -> f32[1,2] {
// CHECK:   ROOT %Arg_0.1 = f32[1,2]{1,0} parameter(0)
// CHECK: }

// -----

///////////////////////////////////////////
// Test Result Layouts                   //
///////////////////////////////////////////

// Test layout
module @entry attributes {
  mhlo.xla_entry_computation_result_layout = [
    dense<[0, 1]> : tensor<2xindex>
  ]
} {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[1,2]{1,0}
// CHECK-SAME: )->f32[1,2]{0,1}}

// CHECK: ENTRY %main.2 (Arg_0.1: f32[1,2]) -> f32[1,2] {
// CHECK:   ROOT %Arg_0.1 = f32[1,2]{1,0} parameter(0)
// CHECK: }

// -----

// Test nested tuple layout
module @entry attributes {
  mhlo.xla_entry_computation_result_layout = [
    [
      [
        dense<[1, 0]> : tensor<2xindex>,
        dense<[0, 1]> : tensor<2xindex>
      ]
    ]
  ]
} {
  func.func @main(
    %arg0: tuple<tuple<tensor<1x2xf32>, tensor<1x2xf32>>>
  ) -> tuple<tuple<tensor<1x2xf32>, tensor<1x2xf32>>> {
    return %arg0 : tuple<tuple<tensor<1x2xf32>, tensor<1x2xf32>>>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   (
// CHECK-SAME:     (
// CHECK-SAME:       f32[1,2]{1,0},
// CHECK-SAME:       f32[1,2]{1,0}
// CHECK-SAME:     )
// CHECK-SAME:   )->((f32[1,2]{1,0}, f32[1,2]{0,1}))
// CHECK-SAME: }

// CHECK: ENTRY %main.2 (Arg_0.1: ((f32[1,2], f32[1,2]))) -> ((f32[1,2], f32[1,2])) {
// CHECK:   ROOT %Arg_0.1 = ((f32[1,2]{1,0}, f32[1,2]{1,0})) parameter(0)
// CHECK: }

// -----

///////////////////////////////////////////
// Test Result Tiles                     //
///////////////////////////////////////////

// Test tile
module @entry attributes {
  mhlo.xla_entry_computation_result_tiles = [
    [dense<[2, 128]> : tensor<2xindex>]
  ]
} {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   f32[1,2]{1,0}
// CHECK-SAME: )->f32[1,2]{1,0:T(2,128)}}

// CHECK: ENTRY %main.2 (Arg_0.1: f32[1,2]) -> f32[1,2] {
// CHECK:   ROOT %Arg_0.1 = f32[1,2]{1,0} parameter(0)
// CHECK: }

// -----

// Test nested tuple tile
module @entry attributes {
  mhlo.xla_entry_computation_result_tiles = [
    [
      [
        [dense<[2, 128]> : tensor<2xindex>],
        [dense<[2, 1]> : tensor<2xindex>]
      ]
    ]
  ]
} {
  func.func @main(
    %arg0: tuple<tuple<tensor<1x2xf32>, tensor<1x2xf32>>>
  ) -> tuple<tuple<tensor<1x2xf32>, tensor<1x2xf32>>> {
    return %arg0 : tuple<tuple<tensor<1x2xf32>, tensor<1x2xf32>>>
  }
}

// CHECK: HloModule entry, entry_computation_layout={(
// CHECK-SAME:   (
// CHECK-SAME:     (
// CHECK-SAME:       f32[1,2]{1,0},
// CHECK-SAME:       f32[1,2]{1,0}
// CHECK-SAME:     )
// CHECK-SAME:   ))->((f32[1,2]{1,0:T(2,128)}, f32[1,2]{1,0:T(2,1)}))
// CHECK-SAME: }

// CHECK: ENTRY %main.2 (Arg_0.1: ((f32[1,2], f32[1,2]))) -> ((f32[1,2], f32[1,2])) {
// CHECK:   ROOT %Arg_0.1 = ((f32[1,2]{1,0}, f32[1,2]{1,0})) parameter(0)
// CHECK: }
