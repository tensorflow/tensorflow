// RUN: sdy_opt %s --split-input-file -xla-sdy-round-trip-export-pipeline | sdy_opt --split-input-file -sdy-round-trip-import-pipeline > t1.mlir
// RUN: sdy_opt %s --split-input-file -xla-sdy-round-trip-export-pipeline | sdy_opt --split-input-file -xla-sdy-round-trip-import-pipeline=enable-constant-import=false > t2.mlir
// RUN: diff t1.mlir t2.mlir

sdy.mesh @mesh_0 = <["axis_0"=2, "axis_1"=4, "axis_2"=4]>
sdy.mesh @mesh_1 = <["axis_0"=16]>
sdy.mesh @mesh_2 = <["x"=8, "y"=4]>

func.func @multiple_shardings(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_2"}, {"axis_0", "axis_1"}]>},
                              %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"axis_0", "axis_2"}]>},
                              %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"axis_1"}]>}) -> tensor<8x16xf32> {
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"axis_1","axis_0"}, {}]>]>} : tensor<8x8xf32>
  %1 = stablehlo.dot %0, %arg2 : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

func.func @multi_result_op(%arg0: tensor<4x64x8xf32>, %arg1: tensor<4x64x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %0 = sdy.constant dense<0.000000e+00> : tensor<f32>
  %1:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {"y"}]>, <@mesh_2, [{"y"}, {}]>]>} :
    (tensor<4x64x8xf32>, tensor<4x64x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<f32>, %arg5: tensor<f32>)  {
      %2 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %3 = stablehlo.add %arg3, %arg5 : tensor<f32>
      stablehlo.return %2, %3 : tensor<f32>, tensor<f32>
    }
  return %1#0, %1#1 : tensor<4x8xf32>, tensor<4x8xf32>
}

func.func @split_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"y"}, {"x":(2)2}]>},
                      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x":(1)2}, {"x":(2)4}]>}) -> tensor<8x16xf32> {
  %1 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x":(1)2, "x":(4)2}, {}]>]>} : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

func.func @func_result_sharding_returning_func_arg(
  %arg0: tensor<8x16xf32>
  ) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", ?}, {"y"}p4]>}) {
  return %arg0 : tensor<8x16xf32>
}

func.func @func_result_sharding_returning_op_value(%arg0: tensor<8x16xf32>)
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", ?}, {"y"}p4]>},
      tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{?}, {"y"}p4]>},
      tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {"y"}p1]>},
      tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  %1:2 = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x","y"}, {}]>, <@mesh_2, [{"y","x"}, {}]>]>} : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
  return %0, %1#0, %1#1, %0 : tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>
}

func.func @sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = sdy.sharding_constraint %arg0 <@mesh_2, [{"x", ?}, {?}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

func.func @export_sharding_group(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  sdy.sharding_group %arg0 group_id = 12:  tensor<8x8xf32>
  return %arg0 : tensor<8x8xf32>
}

func.func @constant() -> tensor<i32> {
  %0 = sdy.constant dense<0> : tensor<i32>
  return %0 : tensor<i32>
}

func.func @inlined_mesh(
  %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a"}]>}
) -> (tensor<32xi32> {sdy.sharding = #sdy.sharding<mesh<[], device_ids=[5]>, []>}) {
  %0 = sdy.sharding_constraint %arg0 <mesh<["c"=4]>, [{"c"}]> : tensor<32xi32>
  return %0 : tensor<32xi32>
}

func.func @op_sharding_rule(%arg0: tensor<8x2xf32>, %arg1: tensor<8x2xf32>) -> tensor<8x2xf64> {
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>} : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf64>
  return %0 : tensor<8x2xf64>
}

func.func @sharding_and_op_sharding_rule(%arg0: tensor<8x2xf32>, %arg1: tensor<8x2xf32>) -> tensor<8x2xf64> {
  %0 = stablehlo.custom_call @foo(%arg0, %arg1)
    {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>,
     sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
    : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf64>
  return %0 : tensor<8x2xf64>
}

func.func @while_with_no_sharding(
    %arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> tensor<32x96xf32> {
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<32> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_1 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare LT, %iterArg_1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    stablehlo.return %iterArg, %iterArg_1 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// -----

func.func @non_sdy_module(%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"},
                          %arg1: tensor<8x8xf32> {mhlo.sharding = "{devices=[1,2,16]<=[32] last_tile_dim_replicate}"},
                          %arg2: tensor<8x16xf32> {mhlo.sharding = "{devices=[4,4,2]<=[2,16]T(1,0) last_tile_dim_replicate}"})
    -> (tensor<8x16xf32> {mhlo.sharding = "{devices=[8,4]<=[32]}"}) {
  %0 = stablehlo.add %arg0, %arg1 {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"} : tensor<8x8xf32>
  %1 = stablehlo.dot %0, %arg2 : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}
