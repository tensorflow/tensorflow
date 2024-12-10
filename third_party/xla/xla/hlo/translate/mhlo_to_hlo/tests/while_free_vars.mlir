// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s -o - | FileCheck %s

// This test verifies that the correct shardings are added when a while loop
// has free variables.

// CHECK-LABEL: HloModule main

// CHECK:      %[[BODY:region_0.*]] ([[ARG_TUPLE:arg_tuple.*]]: (s32[], f32[4], s32[], s32[], f32[4])) -> (s32[], f32[4], s32[], s32[], f32[4]) {
// CHECK-NEXT:   %[[ARG_TUPLE]] = (s32[], f32[4], s32[], s32[], f32[4]) parameter(0)
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {replicated}, {replicated}, {devices=[4]<=[4]}}
// CHECK-DAG:    %[[GTE12:get-tuple-element.*]] = s32[] get-tuple-element((s32[], f32[4], s32[], s32[], f32[4]) %[[ARG_TUPLE]]), index=3
// CHECK-DAG:    %[[GTE13:get-tuple-element.*]] = f32[4] get-tuple-element((s32[], f32[4], s32[], s32[], f32[4]) %[[ARG_TUPLE]]), index=4, sharding={devices=[4]<=[4]}
// CHECK-DAG:    %[[ADD14:add.*]] = s32[] add(s32[] %get-tuple-element.{{.*}}, s32[] %[[GTE12]])
// CHECK-DAG:    %[[ADD15:add.*]] = f32[4] add(f32[4] %get-tuple-element.{{.*}}, f32[4] %[[GTE13]])
// CHECK:        ROOT %tuple.{{.*}} = (s32[], f32[4], s32[], s32[], f32[4]) tuple(s32[] %[[ADD14]], f32[4] %[[ADD15]], s32[] %get-tuple-element.{{.*}}, s32[] %[[GTE12]], f32[4] %[[GTE13]])
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {replicated}, {replicated}, {devices=[4]<=[4]}}

// CHECK:      %[[COND:region_1.*]] ([[ARG_TUPLE:arg_tuple.*]]: (s32[], f32[4], s32[], s32[], f32[4])) -> pred[] {
// CHECK-NEXT:   %[[ARG_TUPLE]] = (s32[], f32[4], s32[], s32[], f32[4]) parameter(0)
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {replicated}, {replicated}, {devices=[4]<=[4]}}
// CHECK:        %[[GTE21:get-tuple-element.*]] = s32[] get-tuple-element((s32[], f32[4], s32[], s32[], f32[4]) %[[ARG_TUPLE]]), index=2
// CHECK-NEXT:   ROOT %compare.{{.*}} = pred[] compare(s32[] %get-tuple-element.{{.*}}, s32[] %[[GTE21]]), direction=LT

// CHECK:      ENTRY %main.{{.*}} ([[ARG0:Arg_0.*]]: s32[], [[ARG1:Arg_1.*]]: f32[4], [[ARG2:Arg_2.*]]: f32[4]) -> f32[4] {
// CHECK-NEXT:   %[[ARG0]] = s32[] parameter(0)
// CHECK-NEXT:   %[[ARG1]] = f32[4] parameter(1)
// CHECK-NEXT:   %[[CONSTANT4:constant.*]] = s32[] constant(0)
// CHECK-NEXT:   %[[CONSTANT5:constant.*]] = s32[] constant(1)
// CHECK-NEXT:   %[[ARG2]] = f32[4] parameter(2)
// CHECK-NEXT:   %[[TUPLE:tuple.*]] = (s32[], f32[4], s32[], s32[], f32[4]) tuple(s32[] %[[ARG0]], f32[4] %[[ARG1]], s32[] %[[CONSTANT4]], s32[] %[[CONSTANT5]], f32[4] %[[ARG2]])
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {replicated}, {replicated}, {devices=[4]<=[4]}}
// CHECK-NEXT:   %[[WHILE:while.25]] = (s32[], f32[4], s32[], s32[], f32[4]) while((s32[], f32[4], s32[], s32[], f32[4]) %[[TUPLE]]), condition=%[[COND]], body=%[[BODY]]
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {replicated}, {replicated}, {devices=[4]<=[4]}}
// CHECK-NEXT:   %[[GTE26:get-tuple-element.*]] = s32[] get-tuple-element((s32[], f32[4], s32[], s32[], f32[4]) %[[WHILE]]), index=0, sharding={replicated}
// CHECK-NEXT:   ROOT %[[GTE27:get-tuple-element.*]] = f32[4] get-tuple-element((s32[], f32[4], s32[], s32[], f32[4]) %[[WHILE]]), index=1, sharding={devices=[2,2]<=[4] last_tile_dim_replicate}

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

// CHECK:      %[[BODY:region_0.*]] ([[ARG_TUPLE:arg_tuple.*]]: (s32[], f32[4], s32[])) -> (s32[], f32[4], s32[]) {
// CHECK-NEXT:   %[[ARG_TUPLE]] = (s32[], f32[4], s32[]) parameter(0)
// CHECK:        %[[GTE:get-tuple-element.*]] = s32[] get-tuple-element((s32[], f32[4], s32[]) %[[ARG_TUPLE]]), index=2
// CHECK:        %[[ADD:add.*]] = s32[] add(s32[] %get-tuple-element.{{.*}}, s32[] %[[GTE]])
// CHECK:        ROOT %tuple.{{.*}} = (s32[], f32[4], s32[]) tuple(s32[] %[[ADD]], f32[4] %get-tuple-element.{{.*}}, s32[] %[[GTE]])

// CHECK:      %[[COND:region_1.*]] ([[ARG_TUPLE:arg_tuple.*]]: (s32[], f32[4], s32[])) -> pred[] {
// CHECK-NEXT:   %[[ARG_TUPLE]] = (s32[], f32[4], s32[]) parameter(0)
// CHECK:        %[[GTE:get-tuple-element..*]] = s32[] get-tuple-element((s32[], f32[4], s32[]) %[[ARG_TUPLE]]), index=2
// CHECK:        ROOT %compare.{{.*}} = pred[] compare(s32[] %get-tuple-element.{{.*}}, s32[] %[[GTE]]), direction=LT

// CHECK:      ENTRY %main.{{.*}} ([[ARG0:Arg_0.*]]: s32[], [[ARG1:Arg_1.*]]: f32[4], [[ARG2:Arg_2.*]]: s32[]) -> f32[4] {
// CHECK-NEXT:   %[[ARG0]] = s32[] parameter(0)
// CHECK-NEXT:   %[[ARG1]] = f32[4] parameter(1)
// CHECK-NEXT:   %[[ARG2]] = s32[] parameter(2)
// CHECK-NEXT:   %[[TUPLE:tuple.*]] = (s32[], f32[4], s32[]) tuple(s32[] %[[ARG0]], f32[4] %[[ARG1]], s32[] %[[ARG2]])
// CHECK-NEXT:   %while.{{.*}} = (s32[], f32[4], s32[]) while((s32[], f32[4], s32[]) %[[TUPLE]]), condition=%[[COND]], body=%[[BODY]]

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
