// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

module {
  func @main(%arg_f32: tensor<4xf32>, %arg_i32: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xi32>, tensor<4xi32>) {
    // CHECK: [[ARG_F32:%.*]] = f32[4] parameter(0)
    // CHECK: [[EXPM1:%.*]] = f32[4] exponential-minus-one(f32[4] [[ARG_F32]])
    %expm1 = "xla_hlo.exponential_minus_one"(%arg_f32) : (tensor<4xf32>) -> tensor<4xf32>

    // CHECK: [[LOG1P:%.*]] = f32[4] log-plus-one(f32[4] [[ARG_F32]])
    %log1p = "xla_hlo.log_plus_one"(%arg_f32) : (tensor<4xf32>) -> tensor<4xf32>

    // CHECK: [[ARG_I32:%.*]] = s32[4] parameter(1)
    // CHECK: [[NOT:%.*]] = s32[4] not(s32[4] [[ARG_I32]])
    %not = "xla_hlo.not"(%arg_i32) : (tensor<4xi32>) -> tensor<4xi32>

    // CHECK: [[POPCNT:%.*]] = s32[4] popcnt(s32[4] [[ARG_I32]])
    %popcnt = "xla_hlo.popcnt"(%arg_i32) : (tensor<4xi32>) -> tensor<4xi32>

    return %expm1, %log1p, %not, %popcnt : tensor<4xf32>, tensor<4xf32>, tensor<4xi32>, tensor<4xi32>
  }
}
