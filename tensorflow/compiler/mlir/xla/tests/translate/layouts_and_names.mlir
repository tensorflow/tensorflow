// RUN: tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text --with-layouts=true --print-layouts=true %s | FileCheck %s

// Checks exporting layouts

// CHECK:  HloModule
func.func @main(%arg0: tensor<128x224x224x4xf16>, %arg1: tensor<64x7x7x4xf16>) -> tensor<128x64x112x112xf16> {
  // CHECK: %convolution.{{.*}} = f16[128,64,112,112]{1,3,2,0} convolution{{.*}}op_name="root.42"
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 0,
      kernel_spatial_dimensions = [1, 2],
      output_batch_dimension = 0,
      output_feature_dimension = 1,
      output_spatial_dimensions = [2, 3]
    >,
    feature_group_count = 1 : i64,
    lhs_dilations = dense<1> : tensor<2xi64>,
    xla_shape = "f16[128,64,112,112]{1,3,2,0}",
    padding = dense<3> : tensor<2x2xi64>,
    precision_config = [ #mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT"> ],
    rhs_dilations = dense<1> : tensor<2xi64>,
    window_strides = dense<2> : tensor<2xi64>
  } : (tensor<128x224x224x4xf16>, tensor<64x7x7x4xf16>)-> tensor<128x64x112x112xf16> loc("root.42")

  // CHECK: s32[1,1]{0,1} constant({ {42} })
  %cst_1 = "arith.constant"() {value = dense<[[42]]> : tensor<1x1xi32>, xla_shape = "s32[1,1]{0,1}"} : () -> tensor<1x1xi32>

  func.return %0 : tensor<128x64x112x112xf16>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: !mhlo.token) -> tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !mhlo.token> {
  %0:3 = "mhlo.infeed"(%arg0) {infeed_config = "foobar", layout=[[0, 1], [0]]} : (!mhlo.token) -> (tensor<3x3xi32>, tensor<i1>, !mhlo.token)
  %1 = "mhlo.tuple"(%0#0, %0#1) : (tensor<3x3xi32>, tensor<i1>) -> tuple<tensor<3x3xi32>, tensor<i1>>
  %2 = "mhlo.tuple"(%1, %0#2) : (tuple<tensor<3x3xi32>, tensor<i1>>, !mhlo.token) -> tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !mhlo.token>

  func.return %2 : tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !mhlo.token>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = token[] parameter(0)
// CHECK:  [[INFEED:%.*]] = ((s32[3,3]{0,1}, pred[]), token[]) infeed(token[] [[ARG]]), infeed_config="foobar"
// CHECK:  [[GTE1:%.*]] = (s32[3,3]{0,1}, pred[]) get-tuple-element(((s32[3,3]{0,1}, pred[]), token[]) [[INFEED]]), index=0
// CHECK:  [[GTE2:%.*]] = s32[3,3]{0,1} get-tuple-element((s32[3,3]{0,1}, pred[]) [[GTE1]]), index=0
// CHECK:  [[GTE3:%.*]] = pred[] get-tuple-element((s32[3,3]{0,1}, pred[]) [[GTE1]]), index=1
// CHECK:  [[GTE4:%.*]] = token[] get-tuple-element(((s32[3,3]{0,1}, pred[]), token[]) [[INFEED]]), index=1

// -----

// CHECK:  HloModule
func.func @main(%arg0: !mhlo.token) -> tuple<tensor<3x3xi32>, !mhlo.token> {
  %0:2 = "mhlo.infeed"(%arg0) {infeed_config = "foobar", layout=[[0,1]]} : (!mhlo.token) -> (tensor<3x3xi32>, !mhlo.token)
  %1 = "mhlo.tuple"(%0#0, %0#1) : (tensor<3x3xi32>, !mhlo.token) -> tuple<tensor<3x3xi32>, !mhlo.token>

  func.return %1 : tuple<tensor<3x3xi32>, !mhlo.token>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = token[] parameter(0)
// CHECK:  [[INFEED:%.*]] = ((s32[3,3]{0,1}), token[]) infeed(token[] [[ARG]]), infeed_config="foobar"
// CHECK:  [[GTE0:%.*]] = (s32[3,3]{0,1}) get-tuple-element(((s32[3,3]{0,1}), token[]) [[INFEED]]), index=0
// CHECK:  [[GTE1:%.*]] = s32[3,3]{0,1} get-tuple-element((s32[3,3]{0,1}) [[GTE0]]), index=0
// CHECK:  [[GTE2:%.*]] = token[] get-tuple-element(((s32[3,3]{0,1}), token[]) [[INFEED]]), index=1
// CHECK:  ROOT [[RES:%.*]] = (s32[3,3]{1,0}, token[]) tuple(s32[3,3]{0,1} [[GTE1]], token[] [[GTE2]]

// -----

// CHECK:  HloModule

func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.infeed"(%arg0) {infeed_config = "foobar", layout = [], xla_shape = "((), token[])"} : (!mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = token[] parameter(0)
// CHECK:  [[INFEED:%.*]] = ((), token[]) infeed(token[] [[ARG]]), infeed_config="foobar"
// CHECK: ROOT [[GTE1:%.*]] = ((), token[]) get-tuple-element(((), token[]) [[INFEED]]), index=1
