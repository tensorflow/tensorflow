// RUN: xla-translate --stablehlo-to-hlo-text -split-input-file %s | FileCheck %s
// RUN: mlir-hlo-opt --stablehlo-legalize-to-hlo=convert-xla-supported-stablehlo=false -split-input-file %s | FileCheck %s --check-prefix CHECK-DIRECT

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[3,4]{1,0}, c64[2]{0}, s32[5]{0}, pred[5]{0}, f32[?,784]{1,0})->(f32[3,4]{1,0}, s32[5]{0}, f32[2]{0}, f32[2]{0}, pred[5]{0}, /*index=5*/f32[?,784]{1,0}, f16[3,4]{1,0}, pred[3,4]{1,0})}

// CHECK:       ENTRY %[[$main_35:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[3,4] parameter(0)
// CHECK-NEXT:  %[[abs_6:[^ ]+]] = f32[3,4] abs(%[[Arg_0_1]]), metadata=
// CHECK-NEXT:  %[[cbrt_7:[^ ]+]] = f32[3,4] cbrt(%[[abs_6]]), metadata=
// CHECK-NEXT:  %[[ceil_8:[^ ]+]] = f32[3,4] ceil(%[[cbrt_7]]), metadata=
// CHECK-NEXT:  %[[cosine_9:[^ ]+]] = f32[3,4] cosine(%[[ceil_8]]), metadata=
// CHECK-NEXT:  %[[exponential_10:[^ ]+]] = f32[3,4] exponential(%[[cosine_9]]), metadata=
// CHECK-NEXT:  %[[exponential_minus_one_11:[^ ]+]] = f32[3,4] exponential-minus-one(%[[exponential_10]]), metadata=
// CHECK-NEXT:  %[[floor_12:[^ ]+]] = f32[3,4] floor(%[[exponential_minus_one_11]]), metadata=
// CHECK-NEXT:  %[[log_13:[^ ]+]] = f32[3,4] log(%[[floor_12]]), metadata=
// CHECK-NEXT:  %[[log_plus_one_14:[^ ]+]] = f32[3,4] log-plus-one(%[[log_13]]), metadata=
// CHECK-NEXT:  %[[logistic_15:[^ ]+]] = f32[3,4] logistic(%[[log_plus_one_14]]), metadata=
// CHECK-NEXT:  %[[negate_16:[^ ]+]] = f32[3,4] negate(%[[logistic_15]]), metadata=
// CHECK-NEXT:  %[[round_nearest_afz_17:[^ ]+]] = f32[3,4] round-nearest-afz(%[[negate_16]]), metadata=
// CHECK-NEXT:  %[[round_nearest_even_18:[^ ]+]] = f32[3,4] round-nearest-even(%[[round_nearest_afz_17]]), metadata=
// CHECK-NEXT:  %[[rsqrt_19:[^ ]+]] = f32[3,4] rsqrt(%[[round_nearest_even_18]]), metadata=
// CHECK-NEXT:  %[[sign_20:[^ ]+]] = f32[3,4] sign(%[[rsqrt_19]]), metadata=
// CHECK-NEXT:  %[[sine_21:[^ ]+]] = f32[3,4] sine(%[[sign_20]]), metadata=
// CHECK-NEXT:  %[[sqrt_22:[^ ]+]] = f32[3,4] sqrt(%[[sine_21]]), metadata=
// CHECK-NEXT:  %[[tan_23:[^ ]+]] = f32[3,4] tan(%[[sqrt_22]]), metadata=
// CHECK-NEXT:  %[[tanh_24:[^ ]+]] = f32[3,4] tanh(%[[tan_23]]), metadata=
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = s32[5] parameter(2)
// CHECK-NEXT:  %[[abs_25:[^ ]+]] = s32[5] abs(%[[Arg_2_3]]), metadata=
// CHECK-NEXT:  %[[count_leading_zeros_26:[^ ]+]] = s32[5] count-leading-zeros(%[[abs_25]]), metadata=
// CHECK-NEXT:  %[[not_27:[^ ]+]] = s32[5] not(%[[count_leading_zeros_26]]), metadata=
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = c64[2] parameter(1)
// CHECK-NEXT:  %[[imag_28:[^ ]+]] = f32[2] imag(%[[Arg_1_2]]), metadata=
// CHECK-NEXT:  %[[real_29:[^ ]+]] = f32[2] real(%[[Arg_1_2]]), metadata=
// CHECK-NEXT:  %[[Arg_3_4:[^ ]+]] = pred[5] parameter(3)
// CHECK-NEXT:  %[[not_30:[^ ]+]] = pred[5] not(%[[Arg_3_4]]), metadata=
// CHECK-NEXT:  %[[Arg_4_5:[^ ]+]] = f32[?,784] parameter(4)
// CHECK-NEXT:  %[[abs_31:[^ ]+]] = f32[?,784] abs(%[[Arg_4_5]]), metadata=
// CHECK-NEXT:  %[[convert_32:[^ ]+]] = f16[3,4] convert(%[[Arg_0_1]]), metadata=
// CHECK-NEXT:  %[[is_finite_33:[^ ]+]] = pred[3,4] is-finite(%[[Arg_0_1]]), metadata=
// CHECK-NEXT:  ROOT %[[tuple_34:[^ ]+]] = (f32[3,4], s32[5], f32[2], f32[2], pred[5], /*index=5*/f32[?,784], f16[3,4], pred[3,4]) tuple(%[[tanh_24]], %[[not_27]], %[[imag_28]], %[[real_29]], %[[not_30]], /*index=5*/%[[abs_31]], %[[convert_32]], %[[is_finite_33]])

func.func @main(
  %arg_f32: tensor<3x4xf32>,
  %arg_complex: tensor<2xcomplex<f32>>,
  %arg_int: tensor<5xi32>,
  %arg_bool: tensor<5xi1>,
  %arg_dynamic: tensor<?x784xf32>
) -> (
  tensor<3x4xf32>,
  tensor<5xi32>,
  tensor<2xf32>,
  tensor<2xf32>,
  tensor<5xi1>,
  tensor<?x784xf32>,
  tensor<3x4xf16>,
  tensor<3x4xi1>
) {
  %f0 = stablehlo.abs %arg_f32 : tensor<3x4xf32>
  %f1 = stablehlo.cbrt %f0 : tensor<3x4xf32>
  %f2 = stablehlo.ceil %f1 : tensor<3x4xf32>
  %f4 = stablehlo.cosine %f2 : tensor<3x4xf32>
  %f6 = stablehlo.exponential %f4 : tensor<3x4xf32>
  %f7 = stablehlo.exponential_minus_one %f6 : tensor<3x4xf32>
  %f8 = stablehlo.floor %f7 : tensor<3x4xf32>
  %f11 = stablehlo.log %f8 : tensor<3x4xf32>
  %f12 = stablehlo.log_plus_one %f11 : tensor<3x4xf32>
  %f13 = stablehlo.logistic %f12 : tensor<3x4xf32>
  %f14 = stablehlo.negate %f13 : tensor<3x4xf32>
  %f19 = stablehlo.round_nearest_afz %f14 : tensor<3x4xf32>
  %f20 = stablehlo.round_nearest_even %f19 : tensor<3x4xf32>
  %f21 = stablehlo.rsqrt %f20 : tensor<3x4xf32>
  %f22 = stablehlo.sign %f21 : tensor<3x4xf32>
  %f23 = stablehlo.sine %f22 : tensor<3x4xf32>
  %f24 = stablehlo.sqrt %f23 : tensor<3x4xf32>
  %f25 = stablehlo.tan %f24 : tensor<3x4xf32>
  %f26 = stablehlo.tanh %f25 : tensor<3x4xf32>
  %i0 = stablehlo.abs %arg_int : tensor<5xi32>
  %i5 = stablehlo.count_leading_zeros %i0 : tensor<5xi32>
  %i16 = stablehlo.not %i5 : tensor<5xi32>
  %cx9 = stablehlo.imag %arg_complex : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  %cx18 = stablehlo.real %arg_complex : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  %b15 = stablehlo.not %arg_bool : tensor<5xi1>
  %d3 = stablehlo.abs %arg_dynamic : tensor<?x784xf32>
  %t3 = stablehlo.convert %arg_f32 : (tensor<3x4xf32>) -> tensor<3x4xf16>
  %t10 = stablehlo.is_finite %arg_f32 : (tensor<3x4xf32>) -> tensor<3x4xi1>

  // Return all the final results to prevent DCE
  func.return %f26, %i16, %cx9, %cx18, %b15, %d3, %t3, %t10 : tensor<3x4xf32>, tensor<5xi32>, tensor<2xf32>, tensor<2xf32>, tensor<5xi1>, tensor<?x784xf32>, tensor<3x4xf16>, tensor<3x4xi1>
}
// CHECK-DIRECT: stablehlo.abs
// CHECK-DIRECT: stablehlo.cbrt
// CHECK-DIRECT: stablehlo.ceil
// CHECK-DIRECT: stablehlo.cosine
// CHECK-DIRECT: stablehlo.exponential
// CHECK-DIRECT: stablehlo.exponential_minus_one
// CHECK-DIRECT: stablehlo.floor
// CHECK-DIRECT: stablehlo.log
// CHECK-DIRECT: stablehlo.log_plus_one
// CHECK-DIRECT: stablehlo.logistic
// CHECK-DIRECT: stablehlo.negate
// CHECK-DIRECT: stablehlo.round_nearest_afz
// CHECK-DIRECT: stablehlo.round_nearest_even
// CHECK-DIRECT: stablehlo.rsqrt
// CHECK-DIRECT: stablehlo.sign
// CHECK-DIRECT: stablehlo.sine
// CHECK-DIRECT: stablehlo.sqrt
// CHECK-DIRECT: stablehlo.tan
// CHECK-DIRECT: stablehlo.tanh
// CHECK-DIRECT: stablehlo.abs
// CHECK-DIRECT: stablehlo.count_leading_zeros
// CHECK-DIRECT: stablehlo.not
// CHECK-DIRECT: stablehlo.imag
// CHECK-DIRECT: stablehlo.real
// CHECK-DIRECT: stablehlo.not
// CHECK-DIRECT: stablehlo.abs
// CHECK-DIRECT: stablehlo.convert
// CHECK-DIRECT: stablehlo.is_finite
