// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s -o - | FileCheck %s

// This test verifies that the correct shardings are added when a while loop
// has free variables.

// CHECK-LABEL: HloModule main

// CHECK:      %region_0.7 (arg_tuple.8: (s32[], f32[4], s32[], s32[], f32[4])) -> (s32[], f32[4], s32[], s32[], f32[4]) {
// CHECK-NEXT:   %arg_tuple.8 = (s32[], f32[4], s32[], s32[], f32[4]) parameter(0)
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {replicated}, {replicated}, {devices=[4]<=[4]}}
// CHECK-DAG:    %get-tuple-element.12 = s32[] get-tuple-element((s32[], f32[4], s32[], s32[], f32[4]) %arg_tuple.8), index=3
// CHECK-DAG:    %get-tuple-element.13 = f32[4] get-tuple-element((s32[], f32[4], s32[], s32[], f32[4]) %arg_tuple.8), index=4, sharding={devices=[4]<=[4]}
// CHECK-DAG:    %add.14 = s32[] add(s32[] %get-tuple-element.9, s32[] %get-tuple-element.12)
// CHECK-DAG:    %add.15 = f32[4] add(f32[4] %get-tuple-element.10, f32[4] %get-tuple-element.13)
// CHECK:        ROOT %tuple.16 = (s32[], f32[4], s32[], s32[], f32[4]) tuple(s32[] %add.14, f32[4] %add.15, s32[] %get-tuple-element.11, s32[] %get-tuple-element.12, f32[4] %get-tuple-element.13)
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {replicated}, {replicated}, {devices=[4]<=[4]}}

// CHECK:      %region_1.17 (arg_tuple.18: (s32[], f32[4], s32[], s32[], f32[4])) -> pred[] {
// CHECK-NEXT:   %arg_tuple.18 = (s32[], f32[4], s32[], s32[], f32[4]) parameter(0)
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {replicated}, {replicated}, {devices=[4]<=[4]}}
// CHECK:        %get-tuple-element.21 = s32[] get-tuple-element((s32[], f32[4], s32[], s32[], f32[4]) %arg_tuple.18), index=2
// CHECK-NEXT:   ROOT %compare.24 = pred[] compare(s32[] %get-tuple-element.19, s32[] %get-tuple-element.21), direction=LT

// CHECK:      ENTRY %main.28 (Arg_0.1: s32[], Arg_1.2: f32[4], Arg_2.3: f32[4]) -> f32[4] {
// CHECK-NEXT:   %Arg_0.1 = s32[] parameter(0)
// CHECK-NEXT:   %Arg_1.2 = f32[4] parameter(1)
// CHECK-NEXT:   %constant.4 = s32[] constant(0)
// CHECK-NEXT:   %constant.5 = s32[] constant(1)
// CHECK-NEXT:   %Arg_2.3 = f32[4] parameter(2)
// CHECK-NEXT:   %tuple.6 = (s32[], f32[4], s32[], s32[], f32[4]) tuple(s32[] %Arg_0.1, f32[4] %Arg_1.2, s32[] %constant.4, s32[] %constant.5, f32[4] %Arg_2.3)
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {replicated}, {replicated}, {devices=[4]<=[4]}}
// CHECK-NEXT:   %while.25 = (s32[], f32[4], s32[], s32[], f32[4]) while((s32[], f32[4], s32[], s32[], f32[4]) %tuple.6), condition=%region_1.17, body=%region_0.7
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {replicated}, {replicated}, {devices=[4]<=[4]}}
// CHECK-NEXT:   %get-tuple-element.26 = s32[] get-tuple-element((s32[], f32[4], s32[], s32[], f32[4]) %while.25), index=0, sharding={replicated}
// CHECK-NEXT:   ROOT %get-tuple-element.27 = f32[4] get-tuple-element((s32[], f32[4], s32[], s32[], f32[4]) %while.25), index=1, sharding={devices=[2,2]<=[4] last_tile_dim_replicate}

func.func @main(%arg0: tensor<i32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32> {mhlo.sharding = "{devices=[4]<=[4]}"}) -> tensor<4xf32> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  %2:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %arg1) : tensor<i32>, tensor<4xf32>
    attributes {mhlo.sharding = "{{replicated},{devices=[2,2]<=[4] last_tile_dim_replicate}}"}
    cond {
    %3 = mhlo.compare LT, %iterArg, %0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    mhlo.return %3 : tensor<i1>
  } do {
    %3 = mhlo.add %iterArg, %1 : tensor<i32>
    %4 = mhlo.add %iterArg_0, %arg2 : tensor<4xf32>
    mhlo.return %3, %4: tensor<i32>, tensor<4xf32>
  }
  func.return %2#1 : tensor<4xf32>
}

// -----

// This test verifies that a value captured multiple times is only lifted once
// and all its uses are replaced. Also verifies that no sharding is added to
// region parameters or root when the while doesn't have a sharding.

// CHECK-LABEL: HloModule main

// CHECK:      %region_0.5 (arg_tuple.6: (s32[], f32[4], s32[])) -> (s32[], f32[4], s32[]) {
// CHECK-NEXT:   %arg_tuple.6 = (s32[], f32[4], s32[]) parameter(0)
// CHECK:        %get-tuple-element.9 = s32[] get-tuple-element((s32[], f32[4], s32[]) %arg_tuple.6), index=2
// CHECK:        %add.10 = s32[] add(s32[] %get-tuple-element.7, s32[] %get-tuple-element.9)
// CHECK:        ROOT %tuple.11 = (s32[], f32[4], s32[]) tuple(s32[] %add.10, f32[4] %get-tuple-element.8, s32[] %get-tuple-element.9)

// CHECK:      %region_1.12 (arg_tuple.13: (s32[], f32[4], s32[])) -> pred[] {
// CHECK-NEXT:   %arg_tuple.13 = (s32[], f32[4], s32[]) parameter(0)
// CHECK:        %get-tuple-element.16 = s32[] get-tuple-element((s32[], f32[4], s32[]) %arg_tuple.13), index=2
// CHECK:        ROOT %compare.17 = pred[] compare(s32[] %get-tuple-element.14, s32[] %get-tuple-element.16), direction=LT

// CHECK:      ENTRY %main.21 (Arg_0.1: s32[], Arg_1.2: f32[4], Arg_2.3: s32[]) -> f32[4] {
// CHECK-NEXT:   %Arg_0.1 = s32[] parameter(0)
// CHECK-NEXT:   %Arg_1.2 = f32[4] parameter(1)
// CHECK-NEXT:   %Arg_2.3 = s32[] parameter(2)
// CHECK-NEXT:   %tuple.4 = (s32[], f32[4], s32[]) tuple(s32[] %Arg_0.1, f32[4] %Arg_1.2, s32[] %Arg_2.3)
// CHECK-NEXT:   %while.18 = (s32[], f32[4], s32[]) while((s32[], f32[4], s32[]) %tuple.4), condition=%region_1.12, body=%region_0.5

func.func @main(%arg0: tensor<i32>, %arg1: tensor<4xf32>, %arg2: tensor<i32>) -> tensor<4xf32> {
  %2:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %arg1) : tensor<i32>, tensor<4xf32>
    cond {
    %3 = mhlo.compare LT, %iterArg, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    mhlo.return %3 : tensor<i1>
  } do {
    %3 = mhlo.add %iterArg, %arg2 : tensor<i32>
    mhlo.return %3, %iterArg_0: tensor<i32>, tensor<4xf32>
  }
  func.return %2#1 : tensor<4xf32>
}
