// RUN: tf-opt -tf-legalize-hlo %s | FileCheck %s --dump-input-on-failure

//===----------------------------------------------------------------------===//
// Binary op legalizations.
//===----------------------------------------------------------------------===//

func @add(%arg0: tensor<2xi32>) -> tensor<2xi32> {
%0 = xla_hlo.add %arg0, %arg0 : tensor<2xi32>
%1 = xla_hlo.add %0, %arg0 : tensor<2xi32>
return %1 : tensor<2xi32>
}
// CHECK-LABEL:   func @add(
// CHECK-SAME:              [[VAL_0:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_1:%.*]] = "tf.AddV2"([[VAL_0]], [[VAL_0]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           [[VAL_2:%.*]] = "tf.AddV2"([[VAL_1]], [[VAL_0]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_2]] : tensor<2xi32>

func @broadcast_add(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
%0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
return %0 : tensor<1x2xi32>
}
// CHECK-LABEL:   func @broadcast_add(
// CHECK-SAME:                        [[VAL_3:%.*]]: tensor<1xi32>, [[VAL_4:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi32> {
// CHECK:           [[VAL_5:%.*]] = "tf.AddV2"([[VAL_3]], [[VAL_4]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           return [[VAL_5]] : tensor<1x2xi32>

func @broadcast_multi_dim_add(%arg0: tensor<4x1x1xi32>, %arg1: tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32> {
%0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<[1, 2, 3]> : tensor<3xi64>} : (tensor<4x1x1xi32>, tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32>
return %0 : tensor<4x4x4x4xi32>
}
// CHECK-LABEL:   func @broadcast_multi_dim_add(
// CHECK-SAME:                                  [[VAL_6:%.*]]: tensor<4x1x1xi32>, [[VAL_7:%.*]]: tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32> {
// CHECK:           [[VAL_8:%.*]] = "tf.AddV2"([[VAL_6]], [[VAL_7]]) : (tensor<4x1x1xi32>, tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32>
// CHECK:           return [[VAL_8]] : tensor<4x4x4x4xi32>

func @div(%arg0: tensor<2xi32>) -> tensor<2xi32> {
%0 = xla_hlo.divide %arg0, %arg0 : tensor<2xi32>
return %0 : tensor<2xi32>
}
// CHECK-LABEL:   func @div(
// CHECK-SAME:              [[VAL_9:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_10:%.*]] = "tf.RealDiv"([[VAL_9]], [[VAL_9]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_10]] : tensor<2xi32>

func @broadcast_div(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
%0 = "xla_hlo.divide"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
return %0 : tensor<1x2xi32>
}
// CHECK-LABEL:   func @broadcast_div(
// CHECK-SAME:                        [[VAL_11:%.*]]: tensor<1xi32>, [[VAL_12:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi32> {
// CHECK:           [[VAL_13:%.*]] = "tf.RealDiv"([[VAL_11]], [[VAL_12]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           return [[VAL_13]] : tensor<1x2xi32>

func @shift_left(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
%0 = xla_hlo.shift_left %arg0, %arg1 : tensor<4xi32>
return %0 : tensor<4xi32>
}
// CHECK-LABEL:   func @shift_left(
// CHECK-SAME:                     [[VAL_14:%.*]]: tensor<4xi32>, [[VAL_15:%.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK:           [[VAL_16:%.*]] = "tf.LeftShift"([[VAL_14]], [[VAL_15]]) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK:           return [[VAL_16]] : tensor<4xi32>

func @div_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
%0 = "xla_hlo.divide"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
return %0 : tensor<?x?xi32>
}
// CHECK-LABEL:   func @div_dynamic(
// CHECK-SAME:                      [[VAL_17:%.*]]: tensor<?xi32>, [[VAL_18:%.*]]: tensor<?x?xi32>) -> tensor<?x?xi32> {
// CHECK:           [[VAL_19:%.*]] = "tf.RealDiv"([[VAL_17]], [[VAL_18]]) : (tensor<?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:           return [[VAL_19]] : tensor<?x?xi32>

func @div_unranked(%arg0: tensor<*xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
%0 = "tf.Div"(%arg0, %arg1) : (tensor<*xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
return %0 : tensor<?x?xi32>
}
// CHECK-LABEL:   func @div_unranked(
// CHECK-SAME:                       [[VAL_20:%.*]]: tensor<*xi32>, [[VAL_21:%.*]]: tensor<?x?xi32>) -> tensor<?x?xi32> {
// CHECK:           [[VAL_22:%.*]] = "tf.Div"([[VAL_20]], [[VAL_21]]) : (tensor<*xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:           return [[VAL_22]] : tensor<?x?xi32>

func @maximum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
%0 = xla_hlo.maximum %arg0, %arg1 : tensor<4xf32>
return %0 : tensor<4xf32>
}
// CHECK-LABEL:   func @maximum(
// CHECK-SAME:                  [[VAL_23:%.*]]: tensor<4xf32>, [[VAL_24:%.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           [[VAL_25:%.*]] = "tf.Maximum"([[VAL_23]], [[VAL_24]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:           return [[VAL_25]] : tensor<4xf32>

func @minimum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
%0 = xla_hlo.minimum %arg0, %arg1 : tensor<4xf32>
return %0 : tensor<4xf32>
}
// CHECK-LABEL:   func @minimum(
// CHECK-SAME:                  [[VAL_26:%.*]]: tensor<4xf32>, [[VAL_27:%.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           [[VAL_28:%.*]] = "tf.Minimum"([[VAL_26]], [[VAL_27]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:           return [[VAL_28]] : tensor<4xf32>

func @mul(%arg0: tensor<2xi32>) -> tensor<2xi32> {
%0 = xla_hlo.multiply %arg0, %arg0 : tensor<2xi32>
return %0 : tensor<2xi32>
}
// CHECK-LABEL:   func @mul(
// CHECK-SAME:              [[VAL_29:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_30:%.*]] = "tf.Mul"([[VAL_29]], [[VAL_29]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_30]] : tensor<2xi32>

func @broadcast_mul(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
%0 = "xla_hlo.multiply"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
return %0 : tensor<1x2xi32>
}
// CHECK-LABEL:   func @broadcast_mul(
// CHECK-SAME:                        [[VAL_31:%.*]]: tensor<1xi32>, [[VAL_32:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi32> {
// CHECK:           [[VAL_33:%.*]] = "tf.Mul"([[VAL_31]], [[VAL_32]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           return [[VAL_33]] : tensor<1x2xi32>

func @real_div(%arg0: tensor<2xi32>) -> tensor<2xi32> {
%0 = xla_hlo.divide %arg0, %arg0 : tensor<2xi32>
return %0 : tensor<2xi32>
}
// CHECK-LABEL:   func @real_div(
// CHECK-SAME:                   [[VAL_34:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_35:%.*]] = "tf.RealDiv"([[VAL_34]], [[VAL_34]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_35]] : tensor<2xi32>

func @broadcast_real_div(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
%0 = "xla_hlo.divide"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
return %0 : tensor<1x2xi32>
}
// CHECK-LABEL:   func @broadcast_real_div(
// CHECK-SAME:                             [[VAL_36:%.*]]: tensor<1xi32>, [[VAL_37:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi32> {
// CHECK:           [[VAL_38:%.*]] = "tf.RealDiv"([[VAL_36]], [[VAL_37]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           return [[VAL_38]] : tensor<1x2xi32>

func @sub(%arg0: tensor<2xi32>) -> tensor<2xi32> {
%0 = xla_hlo.subtract %arg0, %arg0 : tensor<2xi32>
return %0 : tensor<2xi32>
}
// CHECK-LABEL:   func @sub(
// CHECK-SAME:              [[VAL_39:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_40:%.*]] = "tf.Sub"([[VAL_39]], [[VAL_39]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_40]] : tensor<2xi32>

func @broadcast_sub(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
%0 = "xla_hlo.subtract"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
return %0 : tensor<1x2xi32>
}
// CHECK-LABEL:   func @broadcast_sub(
// CHECK-SAME:                        [[VAL_41:%.*]]: tensor<1xi32>, [[VAL_42:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi32> {
// CHECK:           [[VAL_43:%.*]] = "tf.Sub"([[VAL_41]], [[VAL_42]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           return [[VAL_43]] : tensor<1x2xi32>

func @shift_right(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
%0 = xla_hlo.shift_right_arithmetic %arg0, %arg1 : tensor<4xi32>
return %0 : tensor<4xi32>
}
// CHECK-LABEL:   func @shift_right(
// CHECK-SAME:                      [[VAL_44:%.*]]: tensor<4xi32>, [[VAL_45:%.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK:           [[VAL_46:%.*]] = "tf.RightShift"([[VAL_44]], [[VAL_45]]) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK:           return [[VAL_46]] : tensor<4xi32>

func @broadcast_shift_right(%arg0: tensor<4xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x4xi32> {
%0 = "xla_hlo.shift_right_arithmetic"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
return %0 : tensor<2x4xi32>
}
// CHECK-LABEL:   func @broadcast_shift_right(
// CHECK-SAME:                                [[VAL_47:%.*]]: tensor<4xi32>, [[VAL_48:%.*]]: tensor<2x4xi32>) -> tensor<2x4xi32> {
// CHECK:           [[VAL_49:%.*]] = "tf.RightShift"([[VAL_47]], [[VAL_48]]) : (tensor<4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
// CHECK:           return [[VAL_49]] : tensor<2x4xi32>

func @shift_right_unsigned(%arg0: tensor<4xui8>, %arg1: tensor<4xui8>) -> tensor<4xui8> {
%0 = "tf.RightShift"(%arg0, %arg1) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
return %0 : tensor<4xui8>
}
// CHECK-LABEL:   func @shift_right_unsigned(
// CHECK-SAME:                               [[VAL_50:%.*]]: tensor<4xui8>, [[VAL_51:%.*]]: tensor<4xui8>) -> tensor<4xui8> {
// CHECK:           [[VAL_52:%.*]] = "tf.RightShift"([[VAL_50]], [[VAL_51]]) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
// CHECK:           return [[VAL_52]] : tensor<4xui8>

func @broadcast_shift_right_unsigned(%arg0: tensor<4xui8>, %arg1: tensor<2x4xui8>) -> tensor<2x4xui8> {
%0 = "tf.RightShift"(%arg0, %arg1) : (tensor<4xui8>, tensor<2x4xui8>) -> tensor<2x4xui8>
return %0 : tensor<2x4xui8>
}
// CHECK-LABEL:   func @broadcast_shift_right_unsigned(
// CHECK-SAME:                                         [[VAL_53:%.*]]: tensor<4xui8>, [[VAL_54:%.*]]: tensor<2x4xui8>) -> tensor<2x4xui8> {
// CHECK:           [[VAL_55:%.*]] = "tf.RightShift"([[VAL_53]], [[VAL_54]]) : (tensor<4xui8>, tensor<2x4xui8>) -> tensor<2x4xui8>
// CHECK:           return [[VAL_55]] : tensor<2x4xui8>

//===----------------------------------------------------------------------===//
// Unary op legalizations.
//===----------------------------------------------------------------------===//

func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @abs(
// CHECK-SAME:              [[VAL_0:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_1:%.*]] = "tf.Abs"([[VAL_0]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_1]] : tensor<2xf32>

func @abs_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @abs_dynamic(
// CHECK-SAME:                      [[VAL_2:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_3:%.*]] = "tf.Abs"([[VAL_2]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_3]] : tensor<?xf32>

func @abs_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.abs"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @abs_unranked(
// CHECK-SAME:                       [[VAL_4:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_5:%.*]] = "tf.Abs"([[VAL_4]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_5]] : tensor<*xf32>

func @ceil(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.ceil"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @ceil(
// CHECK-SAME:               [[VAL_6:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_7:%.*]] = "tf.Ceil"([[VAL_6]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_7]] : tensor<2xf32>

func @ceil_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.ceil"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @ceil_dynamic(
// CHECK-SAME:                       [[VAL_8:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_9:%.*]] = "tf.Ceil"([[VAL_8]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_9]] : tensor<?xf32>

func @ceil_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.ceil"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @ceil_unranked(
// CHECK-SAME:                        [[VAL_10:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_11:%.*]] = "tf.Ceil"([[VAL_10]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_11]] : tensor<*xf32>

func @cos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.cos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @cos(
// CHECK-SAME:              [[VAL_12:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_13:%.*]] = "tf.Cos"([[VAL_12]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_13]] : tensor<2xf32>

func @cos_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.cos"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @cos_dynamic(
// CHECK-SAME:                      [[VAL_14:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_15:%.*]] = "tf.Cos"([[VAL_14]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_15]] : tensor<?xf32>

func @cos_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.cos"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @cos_unranked(
// CHECK-SAME:                       [[VAL_16:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_17:%.*]] = "tf.Cos"([[VAL_16]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_17]] : tensor<*xf32>

func @exp(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.exp"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @exp(
// CHECK-SAME:              [[VAL_18:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_19:%.*]] = "tf.Exp"([[VAL_18]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_19]] : tensor<2xf32>

func @exp_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.exp"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @exp_dynamic(
// CHECK-SAME:                      [[VAL_20:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_21:%.*]] = "tf.Exp"([[VAL_20]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_21]] : tensor<?xf32>

func @exp_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.exp"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @exp_unranked(
// CHECK-SAME:                       [[VAL_22:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_23:%.*]] = "tf.Exp"([[VAL_22]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_23]] : tensor<*xf32>

func @floor(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @floor(
// CHECK-SAME:                [[VAL_24:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_25:%.*]] = "tf.Floor"([[VAL_24]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_25]] : tensor<2xf32>

func @floor_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.floor"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @floor_dynamic(
// CHECK-SAME:                        [[VAL_26:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_27:%.*]] = "tf.Floor"([[VAL_26]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_27]] : tensor<?xf32>

func @floor_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.floor"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @floor_unranked(
// CHECK-SAME:                         [[VAL_28:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_29:%.*]] = "tf.Floor"([[VAL_28]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_29]] : tensor<*xf32>

func @is_finite(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %0 = "xla_hlo.is_finite"(%arg0) : (tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}
// CHECK-LABEL:   func @is_finite(
// CHECK-SAME:                    [[VAL_30:%.*]]: tensor<2xf32>) -> tensor<2xi1> {
// CHECK:           [[VAL_31:%.*]] = "tf.IsFinite"([[VAL_30]]) : (tensor<2xf32>) -> tensor<2xi1>
// CHECK:           return [[VAL_31]] : tensor<2xi1>

func @is_finite_dynamic(%arg0: tensor<?xf32>) -> tensor<?xi1> {
  %0 = "xla_hlo.is_finite"(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}
// CHECK-LABEL:   func @is_finite_dynamic(
// CHECK-SAME:                            [[VAL_32:%.*]]: tensor<?xf32>) -> tensor<?xi1> {
// CHECK:           [[VAL_33:%.*]] = "tf.IsFinite"([[VAL_32]]) : (tensor<?xf32>) -> tensor<?xi1>
// CHECK:           return [[VAL_33]] : tensor<?xi1>

func @is_finite_unranked(%arg0: tensor<*xf32>) -> tensor<*xi1> {
  %0 = "xla_hlo.is_finite"(%arg0) : (tensor<*xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>
}
// CHECK-LABEL:   func @is_finite_unranked(
// CHECK-SAME:                             [[VAL_34:%.*]]: tensor<*xf32>) -> tensor<*xi1> {
// CHECK:           [[VAL_35:%.*]] = "tf.IsFinite"([[VAL_34]]) : (tensor<*xf32>) -> tensor<*xi1>
// CHECK:           return [[VAL_35]] : tensor<*xi1>

func @log(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.log"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @log(
// CHECK-SAME:              [[VAL_36:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_37:%.*]] = "tf.Log"([[VAL_36]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_37]] : tensor<2xf32>

func @log_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.log"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @log_dynamic(
// CHECK-SAME:                      [[VAL_38:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_39:%.*]] = "tf.Log"([[VAL_38]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_39]] : tensor<?xf32>

func @log_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.log"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @log_unranked(
// CHECK-SAME:                       [[VAL_40:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_41:%.*]] = "tf.Log"([[VAL_40]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_41]] : tensor<*xf32>

func @log1p(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.log_plus_one"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @log1p(
// CHECK-SAME:                [[VAL_42:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_43:%.*]] = "tf.Log1p"([[VAL_42]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_43]] : tensor<2xf32>

func @log1p_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.log_plus_one"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @log1p_dynamic(
// CHECK-SAME:                        [[VAL_44:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_45:%.*]] = "tf.Log1p"([[VAL_44]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_45]] : tensor<?xf32>

func @log1p_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.log_plus_one"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @log1p_unranked(
// CHECK-SAME:                         [[VAL_46:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_47:%.*]] = "tf.Log1p"([[VAL_46]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_47]] : tensor<*xf32>

func @not_op_unranked(%arg0: tensor<*xi1>) -> tensor<*xi1> {
  %0 = "xla_hlo.not"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  return %0 : tensor<*xi1>
}
// CHECK-LABEL:   func @not_op_unranked(
// CHECK-SAME:                          [[VAL_48:%.*]]: tensor<*xi1>) -> tensor<*xi1> {
// CHECK:           [[VAL_49:%.*]] = "tf.LogicalNot"([[VAL_48]]) : (tensor<*xi1>) -> tensor<*xi1>
// CHECK:           return [[VAL_49]] : tensor<*xi1>

func @neg(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.neg"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @neg(
// CHECK-SAME:              [[VAL_50:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_51:%.*]] = "tf.Neg"([[VAL_50]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_51]] : tensor<2xf32>

func @neg_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.neg"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @neg_dynamic(
// CHECK-SAME:                      [[VAL_52:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_53:%.*]] = "tf.Neg"([[VAL_52]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_53]] : tensor<?xf32>

func @neg_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.neg"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @neg_unranked(
// CHECK-SAME:                       [[VAL_54:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_55:%.*]] = "tf.Neg"([[VAL_54]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_55]] : tensor<*xf32>

func @sin(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.sin"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @sin(
// CHECK-SAME:              [[VAL_56:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_57:%.*]] = "tf.Sin"([[VAL_56]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_57]] : tensor<2xf32>

func @sin_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.sin"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @sin_dynamic(
// CHECK-SAME:                      [[VAL_58:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_59:%.*]] = "tf.Sin"([[VAL_58]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_59]] : tensor<?xf32>

func @sin_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.sin"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @sin_unranked(
// CHECK-SAME:                       [[VAL_60:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_61:%.*]] = "tf.Sin"([[VAL_60]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_61]] : tensor<*xf32>

func @rsqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.rsqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @rsqrt(
// CHECK-SAME:                [[VAL_62:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_63:%.*]] = "tf.Rsqrt"([[VAL_62]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_63]] : tensor<2xf32>

func @rsqrt_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.rsqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @rsqrt_dynamic(
// CHECK-SAME:                        [[VAL_64:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_65:%.*]] = "tf.Rsqrt"([[VAL_64]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_65]] : tensor<?xf32>

func @rsqrt_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.rsqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @rsqrt_unranked(
// CHECK-SAME:                         [[VAL_66:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_67:%.*]] = "tf.Rsqrt"([[VAL_66]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_67]] : tensor<*xf32>

func @sqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.sqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @sqrt(
// CHECK-SAME:               [[VAL_68:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_69:%.*]] = "tf.Sqrt"([[VAL_68]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_69]] : tensor<2xf32>

func @sqrt_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.sqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @sqrt_dynamic(
// CHECK-SAME:                       [[VAL_70:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_71:%.*]] = "tf.Sqrt"([[VAL_70]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_71]] : tensor<?xf32>

func @sqrt_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.sqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @sqrt_unranked(
// CHECK-SAME:                        [[VAL_72:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_73:%.*]] = "tf.Sqrt"([[VAL_72]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_73]] : tensor<*xf32>

func @tanh(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.tanh"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL:   func @tanh(
// CHECK-SAME:               [[VAL_74:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_75:%.*]] = "tf.Tanh"([[VAL_74]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_75]] : tensor<2xf32>

func @tanh_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.tanh"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL:   func @tanh_dynamic(
// CHECK-SAME:                       [[VAL_76:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_77:%.*]] = "tf.Tanh"([[VAL_76]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_77]] : tensor<?xf32>

func @tanh_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.tanh"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @tanh_unranked(
// CHECK-SAME:                        [[VAL_78:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_79:%.*]] = "tf.Tanh"([[VAL_78]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_79]] : tensor<*xf32>

