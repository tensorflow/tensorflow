// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: f32[], Arg_1.2: f32[4]) -> f32[4,4]
func.func public @main(%arg0: tensor<f32> {mhlo.sharding = ""}, %arg1: tensor<4xf32> {mhlo.sharding = "\08\03\1A\01\02\22\02\00\01"}) -> (tensor<4x4xf32> {mhlo.sharding = "\08\03\1A\02\02\01\22\02\00\01"}) {
  // CHECK-NEXT: %Arg_1.2 = f32[4] parameter(1), sharding={devices=[2]0,1}
  // CHECK-NEXT: %Arg_0.1 = f32[] parameter(0), sharding={replicated}
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<4xf32>
  %1 = mhlo.multiply %arg1, %0 : tensor<4xf32>
  %2 = "mhlo.broadcast_in_dim"(%1) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<4xf32>) -> tensor<4x4xf32>
  // CHECK: ROOT {{.*}}, sharding={devices=[2,1]0,1}
  func.return %2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: ENTRY %main.{{.*}} ({{[^,]*}}: f32[5,8,128]) -> f32[5,8,128]
func.func @main(%arg0: tensor<5x8x128xf32> {mhlo.sharding = "\08\03\1A\03\01\02\01\22\02\00\01"}) -> (tensor<5x8x128xf32> {mhlo.sharding = "\08\03\1A\03\01\02\01\22\02\00\01"}) {
  // CHECK-NEXT: %Arg_0.1 = f32[5,8,128] parameter(0), sharding={devices=[1,2,1]0,1}
  // CHECK-NEXT: %custom-call.2 = f32[5,8,128] custom-call(f32[5,8,128] %Arg_0.1), custom_call_target="Sharding", sharding={devices=[1,2,1]0,1}
  // CHECK-NEXT: %tuple.3 = (f32[5,8,128]) tuple(f32[5,8,128] %custom-call.2)
  // CHECK-NEXT: ROOT %get-tuple-element.4 = f32[5,8,128] get-tuple-element((f32[5,8,128]) %tuple.3), index=0
  // CHECK-SAME: sharding={devices=[1,2,1]0,1}
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "Sharding",
				  mhlo.sharding = "\08\03\1A\03\01\02\01\22\02\00\01"
				 } : (tensor<5x8x128xf32>) -> tensor<5x8x128xf32>
  func.return %0 : tensor<5x8x128xf32>
}

// -----

// CHECK-LABEL: ENTRY %main.{{.*}} ({{[^,]*}}: f32[4,4]) -> (f32[4,4], f32[4,4])
func.func @main(%arg0: tensor<4x4xf32>) -> (tensor<4x4xf32> {mhlo.sharding = "\08\03\1A\03\02\01\02\22\04\00\01\02\03B\01\00"}, tensor<4x4xf32>) {
  // CHECK-NEXT: %Arg_0.1 = f32[4,4] parameter(0)
  // CHECK-NEXT: [[RESHAPE_0:%.*]] = f32[4,4] reshape(f32[4,4] %Arg_0.1), sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  // CHECK-NEXT: [[RESHAPE_1:%.*]] = f32[4,4] reshape(f32[4,4] %Arg_0.1)
  // CHECK-NOT:  sharding
  // CHECK-NEXT: ROOT {{%.*}} = (f32[4,4], f32[4,4]) tuple(f32[4,4] [[RESHAPE_0]], f32[4,4] [[RESHAPE_1]])
  // CHECK-SAME: sharding={{\{}}{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}, {replicated}}
  return %arg0, %arg0 : tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// CHECK-LABEL: ENTRY %main.{{.*}} () -> f32[4]
func.func @main() -> (tensor<4xf32>) {
  // CHECK-NEXT: %constant.1 = f32[] constant(3.1415925)
  // CHECK-NEXT: %broadcast.2 = f32[4] broadcast(f32[] %constant.1), dimensions={}, sharding={devices=[2]0,1}
  // CHECK-NEXT: ROOT %add.3 = f32[4] add(f32[4] %broadcast.2, f32[4] %broadcast.2)
  %0 = mhlo.constant {mhlo.sharding = "{devices=[2]0,1}"} dense<3.1415926> : tensor<4xf32>
  %1 = mhlo.add %0, %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: ENTRY %main.{{.*}} () -> f32[12,24,36]
func.func @main() -> (tensor<12x24x36xf32>) {
  // CHECK-NEXT: %constant.1 = f32[] constant(3.1415925)
  // CHECK-NEXT: %broadcast.2 = f32[12,24,36] broadcast(f32[] %constant.1), dimensions={}, sharding={devices=[1,2,1]0,1}
  // CHECK-NEXT: ROOT %add.3 = f32[12,24,36] add(f32[12,24,36] %broadcast.2, f32[12,24,36] %broadcast.2)
  %0 = mhlo.constant {mhlo.sharding = "{devices=[1,2,1]0,1}"} dense<3.1415926> : tensor<12x24x36xf32>
  %1 = mhlo.add %0, %0 : tensor<12x24x36xf32>
  return %1 : tensor<12x24x36xf32>
}

// -----

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: u64[2]) -> (u64[2], u32[512,4])
func.func @main(%arg0: tensor<2xui64>) -> (tensor<2xui64> {mhlo.sharding = "{devices=[2,16]<=[32] last_tile_dim_replicate}"}, tensor<512x4xui32> {mhlo.sharding = "{devices=[4,8]<=[32]}"}) {
  // CHECK-NEXT: %Arg_0.1 = u64[2] parameter(0)
  // CHECK-NEXT: %rng-bit-generator.2 = (u64[2], u32[512,4]) rng-bit-generator(u64[2] %Arg_0.1), algorithm=rng_default, sharding={{\{}}{replicated}, {devices=[8,4]<=[32]}}
  // CHECK-NEXT: %get-tuple-element.3 = u64[2] get-tuple-element((u64[2], u32[512,4]) %rng-bit-generator.2), index=0, sharding={replicated}
  // CHECK-NEXT: %add.5 = u64[2] add(u64[2] %get-tuple-element.3, u64[2] %get-tuple-element.3)
  // CHECK-NEXT: %reshape.6 = u64[2] reshape(u64[2] %add.5)
  // CHECK-NEXT: %get-tuple-element.4 = u32[512,4] get-tuple-element((u64[2], u32[512,4]) %rng-bit-generator.2), index=1, sharding={devices=[8,4]<=[32]}
  // CHECK-NEXT: %reshape.7 = u32[512,4] reshape(u32[512,4] %get-tuple-element.4)
  // CHECK-NEXT: ROOT %tuple.8 = (u64[2], u32[512,4]) tuple(u64[2] %reshape.6, u32[512,4] %reshape.7), sharding={{\{}}{devices=[2,16]<=[32] last_tile_dim_replicate}, {devices=[4,8]<=[32]}}
  %output_state, %output = "mhlo.rng_bit_generator"(%arg0) <{rng_algorithm = #mhlo.rng_algorithm<DEFAULT>}> {mhlo.sharding = "{{replicated}, {devices=[8,4]<=[32]}}"} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<512x4xui32>)
  %0 = mhlo.add %output_state, %output_state : tensor<2xui64>
  return %0, %output : tensor<2xui64>, tensor<512x4xui32>
}

// -----

// CHECK-LABEL: HloModule main

// CHECK:      %region_0.2 (Arg_.3: s32[]) -> s32[] {
// CHECK-NEXT:   %Arg_.3 = s32[] parameter(0), sharding={replicated}
// CHECK-NEXT:   %add.4 = s32[] add(s32[] %Arg_.3, s32[] %Arg_.3)
// CHECK-NEXT:   %tuple.5 = (s32[]) tuple(s32[] %add.4)
// CHECK-NEXT:   ROOT %get-tuple-element.6 = s32[] get-tuple-element((s32[]) %tuple.5), index=0, sharding={replicated}

// CHECK:      %region_1.7 (Arg_.8: s32[]) -> pred[] {
// CHECK-NEXT:   %Arg_.8 = s32[] parameter(0), sharding={replicated}
// CHECK-NEXT:   ROOT %compare.9 = pred[] compare(s32[] %Arg_.8, s32[] %Arg_.8), direction=LT

// CHECK:      ENTRY %main.11 (Arg_0.1: s32[]) -> s32[] {
// CHECK-NEXT:   %Arg_0.1 = s32[] parameter(0)
// CHECK-NEXT:   ROOT %while.10 = s32[] while(s32[] %Arg_0.1), condition=%region_1.7, body=%region_0.2, sharding={replicated}

func.func @main(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = mhlo.while(%iterArg = %arg0) : tensor<i32> attributes {mhlo.sharding = "{replicated}"}
    cond {
    %1 = mhlo.compare LT, %iterArg, %iterArg : (tensor<i32>, tensor<i32>) -> tensor<i1>
    mhlo.return %1 : tensor<i1>
  } do {
    %1 = mhlo.add %iterArg, %iterArg : tensor<i32>
    mhlo.return %1 : tensor<i32>
  }
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: HloModule main

// CHECK:      %region_0.5 (arg_tuple.6: (s32[], f32[4], f32[4])) -> (s32[], f32[4], f32[4]) {
// CHECK-NEXT:   %arg_tuple.6 = (s32[], f32[4], f32[4]) parameter(0)
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {devices=[4]<=[4]}}
// CHECK-NEXT:   %get-tuple-element.7 = s32[] get-tuple-element((s32[], f32[4], f32[4]) %arg_tuple.6), index=0, sharding={replicated}
// CHECK-NEXT:   %get-tuple-element.8 = f32[4] get-tuple-element((s32[], f32[4], f32[4]) %arg_tuple.6), index=1, sharding={devices=[2,2]<=[4] last_tile_dim_replicate}
// CHECK-NEXT:   %get-tuple-element.9 = f32[4] get-tuple-element((s32[], f32[4], f32[4]) %arg_tuple.6), index=2, sharding={devices=[4]<=[4]}
// CHECK-NEXT:   %add.10 = f32[4] add(f32[4] %get-tuple-element.8, f32[4] %get-tuple-element.9)
// CHECK-NEXT:   ROOT %tuple.11 = (s32[], f32[4], f32[4]) tuple(s32[] %get-tuple-element.7, f32[4] %add.10, f32[4] %get-tuple-element.9)
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {devices=[4]<=[4]}}

// CHECK:      %region_1.12 (arg_tuple.13: (s32[], f32[4], f32[4])) -> pred[] {
// CHECK-NEXT:   %arg_tuple.13 = (s32[], f32[4], f32[4]) parameter(0)
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {devices=[4]<=[4]}}
// CHECK-NEXT:   %get-tuple-element.15 = f32[4] get-tuple-element((s32[], f32[4], f32[4]) %arg_tuple.13), index=1, sharding={devices=[2,2]<=[4] last_tile_dim_replicate}
// CHECK-NEXT:   %get-tuple-element.16 = f32[4] get-tuple-element((s32[], f32[4], f32[4]) %arg_tuple.13), index=2, sharding={devices=[4]<=[4]}
// CHECK-NEXT:   %get-tuple-element.14 = s32[] get-tuple-element((s32[], f32[4], f32[4]) %arg_tuple.13), index=0, sharding={replicated}
// CHECK-NEXT:   ROOT %compare.17 = pred[] compare(s32[] %get-tuple-element.14, s32[] %get-tuple-element.14), direction=LT

// CHECK:      ENTRY %main.23 (Arg_0.1: s32[], Arg_1.2: f32[4], Arg_2.3: f32[4]) -> (f32[4], f32[4]) {
// CHECK-NEXT:   %Arg_0.1 = s32[] parameter(0)
// CHECK-NEXT:   %Arg_1.2 = f32[4] parameter(1)
// CHECK-NEXT:   %Arg_2.3 = f32[4] parameter(2)
// CHECK-NEXT:   %tuple.4 = (s32[], f32[4], f32[4]) tuple(s32[] %Arg_0.1, f32[4] %Arg_1.2, f32[4] %Arg_2.3)
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {devices=[4]<=[4]}}
// CHECK-NEXT:   %while.18 = (s32[], f32[4], f32[4]) while((s32[], f32[4], f32[4]) %tuple.4), condition=%region_1.12, body=%region_0.5
// CHECK-SAME:     sharding={{\{}}{replicated}, {devices=[2,2]<=[4] last_tile_dim_replicate}, {devices=[4]<=[4]}}
// CHECK-NEXT:   %get-tuple-element.19 = s32[] get-tuple-element((s32[], f32[4], f32[4]) %while.18), index=0, sharding={replicated}
// CHECK-NEXT:   %get-tuple-element.20 = f32[4] get-tuple-element((s32[], f32[4], f32[4]) %while.18), index=1, sharding={devices=[2,2]<=[4] last_tile_dim_replicate}
// CHECK-NEXT:   %get-tuple-element.21 = f32[4] get-tuple-element((s32[], f32[4], f32[4]) %while.18), index=2, sharding={devices=[4]<=[4]}
// CHECK-NEXT:   ROOT %tuple.22 = (f32[4], f32[4]) tuple(f32[4] %get-tuple-element.20, f32[4] %get-tuple-element.21)

func.func @main(%arg0: tensor<i32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0:3 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %arg1, %iterArg_1 = %arg2) : tensor<i32>, tensor<4xf32>, tensor<4xf32>
    attributes {mhlo.sharding = "{{replicated},{devices=[2,2]<=[4] last_tile_dim_replicate},{devices=[4]<=[4]}}"}
    cond {
    %1 = mhlo.compare LT, %iterArg, %iterArg : (tensor<i32>, tensor<i32>) -> tensor<i1>
    mhlo.return %1 : tensor<i1>
  } do {
    %1 = mhlo.add %iterArg_0, %iterArg_1 : tensor<4xf32>
    mhlo.return %iterArg, %1, %iterArg_1 : tensor<i32>, tensor<4xf32>, tensor<4xf32>
  }
  func.return %0#1, %0#2 : tensor<4xf32>, tensor<4xf32>
}
