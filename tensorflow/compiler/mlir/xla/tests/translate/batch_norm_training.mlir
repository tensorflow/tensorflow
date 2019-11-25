// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>> {
  %0 = "xla_hlo.batch_norm_training" (%input, %scale, %offset) {epsilon = 0.001 : f32, feature_index = 3 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
  return %0 : tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
}

// CHECK-LABEL:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[2,2,2,2] parameter(0)
// CHECK:  [[VAL_2:%.*]] = f32[2] parameter(1)
// CHECK:  [[VAL_3:%.*]] = f32[2] parameter(2)
// CHECK-LABEL:  ROOT
// CHECK-SAME:  (f32[2,2,2,2], f32[2], f32[2]) batch-norm-training(f32[2,2,2,2] [[VAL_1]], f32[2] [[VAL_2]], f32[2] [[VAL_3]]), epsilon=0.001, feature_index=3
