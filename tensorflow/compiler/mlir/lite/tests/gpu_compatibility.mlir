// RUN: litert-opt %s -tfl-gpu-compatibility | FileCheck %s

module {
    func.func @test_strided_slice_promotion(%arg0: tensor<1x12xi1>) -> tensor<1x3xi1> {
    // CHECK-LABEL: func.func @test_strided_slice_promotion
    // CHECK-DAG: %[[C0:.*]] = "tfl.pseudo_const"() <{value = dense<0> : tensor<2xi32>}>
    // CHECK-DAG: %[[C1:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 3]> : tensor<2xi32>}>
    // CHECK-DAG: %[[C2:.*]] = "tfl.pseudo_const"() <{value = dense<1> : tensor<2xi32>}>
    // CHECK: %[[CAST_IN:.*]] = "tfl.cast"(%arg0) : (tensor<1x12xi1>) -> tensor<1x12xf32>
    // CHECK: %[[SS:.*]] = "tfl.strided_slice"(%[[CAST_IN]], %[[C0]], %[[C1]], %[[C2]])
    // CHECK-SAME: {{.*}} (tensor<1x12xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x3xf32>
    // CHECK: %[[CAST_OUT:.*]] = "tfl.cast"(%[[SS]]) : (tensor<1x3xf32>) -> tensor<1x3xi1>
    // CHECK: return %[[CAST_OUT]]
    %cst = "tfl.pseudo_const"() <{value = dense<[0, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %cst_0 = "tfl.pseudo_const"() <{value = dense<[1, 3]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %cst_1 = "tfl.pseudo_const"() <{value = dense<[1, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<1x12xi1>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x3xi1>
    func.return %0 : tensor<1x3xi1>
  }

  func.func @test_pad_promotion_expansion(%arg0: tensor<1x36xi1>) -> tensor<1x48xi1> {
    // CHECK-LABEL: func.func @test_pad_promotion_expansion
    // CHECK-DAG: %[[C0:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 48]> : tensor<2xi32>}>
    // CHECK-DAG: %[[C1:.*]] = "tfl.pseudo_const"() <{value = {{.*}} : tensor<3x2xi32>}>
    // CHECK-DAG: %[[C2:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 1, 36]> : tensor<3xi32>}>
    // CHECK: %[[RE_IN:.*]] = "tfl.reshape"(%arg0, %[[C2]]) : (tensor<1x36xi1>, tensor<3xi32>) -> tensor<1x1x36xi1>
    // CHECK: %[[CAST_IN:.*]] = "tfl.cast"(%[[RE_IN]]) : (tensor<1x1x36xi1>) -> tensor<1x1x36xf32>
    // CHECK: %[[PAD:.*]] = "tfl.pad"(%[[CAST_IN]], %[[C1]]) : (tensor<1x1x36xf32>, tensor<3x2xi32>) -> tensor<1x1x48xf32>
    // CHECK: %[[CAST_OUT:.*]] = "tfl.cast"(%[[PAD]]) : (tensor<1x1x48xf32>) -> tensor<1x1x48xi1>
    // CHECK: %[[RE_OUT:.*]] = "tfl.reshape"(%[[CAST_OUT]], %[[C0]]) : (tensor<1x1x48xi1>, tensor<2xi32>) -> tensor<1x48xi1>
    // CHECK: return %[[RE_OUT]]
    %cst = "tfl.pseudo_const"() <{value = dense<[[0, 0], [0, 12]]> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
    %0 = "tfl.pad"(%arg0, %cst) : (tensor<1x36xi1>, tensor<2x2xi32>) -> tensor<1x48xi1>
    func.return %0 : tensor<1x48xi1>
  }

  func.func @test_batch_matmul_swap(%arg0: tensor<8x48x24xf32>) -> tensor<8x13x24xf32> {
    // CHECK-LABEL: func.func @test_batch_matmul_swap
    // CHECK-DAG: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<1.000000e+00> : tensor<8x13x48xf32>}> : () -> tensor<8x13x48xf32>
    // CHECK-DAG: %[[PERM:.*]] = "tfl.pseudo_const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    // CHECK: %[[BMM:.*]] = "tfl.batch_matmul"(%arg0, %[[CST]]) <{adj_x = true, adj_y = true, asymmetric_quantize_inputs = false}> : (tensor<8x48x24xf32>, tensor<8x13x48xf32>) -> tensor<8x24x13xf32>
    // CHECK: %[[RES:.*]] = "tfl.transpose"(%[[BMM]], %[[PERM]]) : (tensor<8x24x13xf32>, tensor<3xi32>) -> tensor<8x13x24xf32>
    // CHECK: return %[[RES]] : tensor<8x13x24xf32>
    %cst = "tfl.pseudo_const"() <{value = dense<1.0> : tensor<8x13x48xf32>}> : () -> tensor<8x13x48xf32>
    %0 = "tfl.batch_matmul"(%cst, %arg0) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<8x13x48xf32>, tensor<8x48x24xf32>) -> tensor<8x13x24xf32>
    func.return %0 : tensor<8x13x24xf32>
  }

  func.func @test_broadcast_to(%arg0: tensor<1x2x1x24xi1>, %arg1: tensor<1x1x12x24xi1>) -> tensor<1x2x12x24xi1> {
    // CHECK-LABEL: func.func @test_broadcast_to
    // CHECK-DAG: %[[MULT2:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 2, 1, 1]> : tensor<4xi64>}> : () -> tensor<4xi64>
    // CHECK-DAG: %[[MULT1:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 1, 12, 1]> : tensor<4xi64>}> : () -> tensor<4xi64>
    // CHECK: %[[CAST1:.*]] = "tfl.cast"(%arg0) : (tensor<1x2x1x24xi1>) -> tensor<1x2x1x24xi32>
    // CHECK: %[[TILE1:.*]] = "tfl.tile"(%[[CAST1]], %[[MULT1]]) : (tensor<1x2x1x24xi32>, tensor<4xi64>) -> tensor<1x2x12x24xi32>
    // CHECK: %[[CAST2:.*]] = "tfl.cast"(%[[TILE1]]) : (tensor<1x2x12x24xi32>) -> tensor<1x2x12x24xi1>
    // CHECK: %[[CAST3:.*]] = "tfl.cast"(%arg1) : (tensor<1x1x12x24xi1>) -> tensor<1x1x12x24xi32>
    // CHECK: %[[TILE2:.*]] = "tfl.tile"(%[[CAST3]], %[[MULT2]]) : (tensor<1x1x12x24xi32>, tensor<4xi64>) -> tensor<1x2x12x24xi32>
    // CHECK: %[[CAST4:.*]] = "tfl.cast"(%[[TILE2]]) : (tensor<1x2x12x24xi32>) -> tensor<1x2x12x24xi1>
    // CHECK: %[[RES:.*]] = tfl.logical_and %[[CAST2]], %[[CAST4]] : tensor<1x2x12x24xi1>
    // CHECK: return %[[RES]] : tensor<1x2x12x24xi1>
    %0 = "tfl.pseudo_const"() <{value = dense<[1, 2, 12, 24]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %1 = "tfl.broadcast_to"(%arg0, %0) : (tensor<1x2x1x24xi1>, tensor<4xi64>) -> tensor<1x2x12x24xi1>
    %2 = "tfl.broadcast_to"(%arg1, %0) : (tensor<1x1x12x24xi1>, tensor<4xi64>) -> tensor<1x2x12x24xi1>
    %3 = tfl.logical_and %1, %2 : tensor<1x2x12x24xi1>
    func.return %3 : tensor<1x2x12x24xi1>
  }

  func.func @test_broadcast_sum_to_mul(%arg0: tensor<25x1x1x1xf32>) -> tensor<25x1x1x1xf32> {
    // CHECK-LABEL: func.func @test_broadcast_sum_to_mul
    // CHECK: %[[SCALE:.*]] = "tfl.pseudo_const"() <{value = dense<8.192000e+03> : tensor<f32>}> : () -> tensor<f32>
    // CHECK: %[[RES:.*]] = tfl.mul(%arg0, %[[SCALE]]) <{fused_activation_function = "NONE"}> : (tensor<25x1x1x1xf32>, tensor<f32>) -> tensor<25x1x1x1xf32>
    // CHECK: return %[[RES]] : tensor<25x1x1x1xf32>
    %0 = "tfl.pseudo_const"() <{value = dense<[25, 64, 1, 128]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %1 = "tfl.broadcast_to"(%arg0, %0) : (tensor<25x1x1x1xf32>, tensor<4xi64>) -> tensor<25x64x1x128xf32>
    %2 = "tfl.pseudo_const"() <{value = dense<[1, 3]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3 = "tfl.sum"(%1, %2) <{keep_dims = true}> : (tensor<25x64x1x128xf32>, tensor<2xi32>) -> tensor<25x1x1x1xf32>
    func.return %3 : tensor<25x1x1x1xf32>
  }

  func.func @test_fuse_dequantize_fc(%arg0: tensor<1x16x384xf32>, %arg1: tensor<1536x384x!quant.uniform<i4:f32:0, {1.0}>>) -> tensor<1x16x1536xf32> {
    // CHECK-LABEL: func.func @test_fuse_dequantize_fc
    // CHECK: %[[CST:.*]] = "tfl.no_value"()
    // CHECK: %[[FC:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[CST]])
    // CHECK-SAME: asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"
    // CHECK-NEXT: return %[[FC]]
    %cst = "tfl.no_value"() {value} : () -> none
    %0 = "tfl.dequantize"(%arg1) : (tensor<1536x384x!quant.uniform<i4:f32:0, {1.0}>>) -> tensor<1536x384xf32>
    %1 = "tfl.fully_connected"(%arg0, %0, %cst) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<1x16x384xf32>, tensor<1536x384xf32>, none) -> tensor<1x16x1536xf32>
    return %1 : tensor<1x16x1536xf32>
  }

  func.func @test_swap_add(%arg0: tensor<f32>, %arg1: tensor<25x1x1x1xf32>) -> tensor<25x1x1x1xf32> {
    // CHECK-LABEL: func.func @test_swap_add
    // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    // CHECK: %[[RESHAPE:.*]] = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<f32>, tensor<1xi32>) -> tensor<1xf32>
    // CHECK: tfl.add(%arg1, %[[RESHAPE]]) <{fused_activation_function = "NONE"}> : (tensor<25x1x1x1xf32>, tensor<1xf32>) -> tensor<25x1x1x1xf32>
    %0 = "tfl.add"(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<f32>, tensor<25x1x1x1xf32>) -> tensor<25x1x1x1xf32>
    func.return %0 : tensor<25x1x1x1xf32>
  }

  func.func @test_tile_non_splat(%arg0: tensor<1x1x12x24xi1>) -> (tensor<1x2x12x24xi1>, tensor<1x2x2xi32>) {
    // CHECK-LABEL: func.func @test_tile_non_splat
    // CHECK: %[[CST0:.*]] = arith.constant dense<true> : tensor<1x2x12x24xi1>
    // CHECK: %[[CST1:.*]] = "tfl.pseudo_const"() <{value = dense<{{.*}}> : tensor<1x2x2xi32>}>
    // CHECK: return %[[CST0]], %[[CST1]]
    %cst = "tfl.pseudo_const"() <{value = dense<true> : tensor<1x1x12x24xi1>}> : () -> tensor<1x1x12x24xi1>
    %multiples = "tfl.pseudo_const"() <{value = dense<[1, 2, 1, 1]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %0 = "tfl.tile"(%cst, %multiples) : (tensor<1x1x12x24xi1>, tensor<4xi64>) -> tensor<1x2x12x24xi1>

    %cst_non_splat = "tfl.pseudo_const"() <{value = dense<[[[1, 2], [3, 4]]]> : tensor<1x2x2xi32>}> : () -> tensor<1x2x2xi32>
    %multiples_2 = "tfl.pseudo_const"() <{value = dense<[1, 2, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %1 = "tfl.tile"(%cst_non_splat, %multiples_2) : (tensor<1x2x2xi32>, tensor<3xi64>) -> tensor<1x2x2xi32>

    return %0, %1 : tensor<1x2x12x24xi1>, tensor<1x2x2xi32>
  }

  func.func @tile_bool(%arg0: tensor<1x2x1x24xi1>, %arg1: tensor<4xi64>) -> tensor<1x2x12x24xi1> {
    // CHECK-LABEL: tile_bool
    // CHECK: %[[CAST_IN:.*]] = "tfl.cast"(%arg0) : (tensor<1x2x1x24xi1>) -> tensor<1x2x1x24xi32>
    // CHECK: %[[TILE:.*]] = "tfl.tile"(%[[CAST_IN]], %arg1) : (tensor<1x2x1x24xi32>, tensor<4xi64>) -> tensor<1x2x12x24xi32>
    // CHECK: %[[CAST_OUT:.*]] = "tfl.cast"(%[[TILE]]) : (tensor<1x2x12x24xi32>) -> tensor<1x2x12x24xi1>
    // CHECK: return %[[CAST_OUT]]
    %0 = "tfl.tile"(%arg0, %arg1) : (tensor<1x2x1x24xi1>, tensor<4xi64>) -> tensor<1x2x12x24xi1>
    func.return %0 : tensor<1x2x12x24xi1>
  }
}
