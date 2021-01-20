// RUN: mlir-hlo-opt --chlo-legalize-to-hlo --split-input-file %s | FileCheck %s

// CHECK-LABEL: @asinh_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func @asinh_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  // Check for the bf16-specific max value.
  // CHECK: mhlo.constant dense<3.389{{.*}}e+38>
  %result = "chlo.asinh"(%arg) : (tensor<bf16>) -> tensor<bf16>
  return %result : tensor<bf16>
}

// CHECK-LABEL: @asinh_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func @asinh_f16(%arg : tensor<f16>) -> tensor<f16> {
  // Check for the f16-specific max value.
  // CHECK: mhlo.constant dense<6.550{{.*}}e+04>
  %result = "chlo.asinh"(%arg) : (tensor<f16>) -> tensor<f16>
  return %result : tensor<f16>
}

// CHECK-LABEL: @asinh_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func @asinh_f32(%arg : tensor<f32>) -> tensor<f32> {
  // Check for the f32-specific max value.
  // CHECK: mhlo.constant dense<3.402{{.*}}E+38>
  %result = "chlo.asinh"(%arg) : (tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}

// CHECK-LABEL: @asinh_f64
// CHECK-SAME: %[[ARG:.*]]: tensor<f64>
func @asinh_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK: %[[TMP_0:.*]] = "mhlo.sign"(%[[ARG]])
  // CHECK: %[[TMP_1:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_2:.*]] = mhlo.constant dense<1.797{{.*}}E+308>
  // CHECK: %[[TMP_3:.*]] = "mhlo.sqrt"(%[[TMP_2]])
  // CHECK: %[[TMP_4:.*]] = "mhlo.compare"(%[[TMP_1]], %[[TMP_3]]) {comparison_direction = "GE"}
  // CHECK: %[[TMP_5:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_6:.*]] = "mhlo.log"(%[[TMP_5]])
  // CHECK: %[[TMP_7:.*]] = mhlo.constant dense<2.000{{.*}}e+00>
  // CHECK: %[[TMP_8:.*]] = "mhlo.log"(%[[TMP_7]])
  // CHECK: %[[TMP_9:.*]] = mhlo.add %[[TMP_6]], %[[TMP_8]]
  // CHECK: %[[TMP_10:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_11:.*]] = mhlo.constant dense<1.000{{.*}}e+00>
  // CHECK: %[[TMP_12:.*]] = "mhlo.compare"(%[[TMP_10]], %[[TMP_11]]) {comparison_direction = "LE"}
  // CHECK: %[[TMP_13:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_14:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_15:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_16:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_17:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_18:.*]] = mhlo.multiply %[[TMP_16]], %[[TMP_17]]
  // CHECK: %[[TMP_19:.*]] = mhlo.constant dense<1.000{{.*}}e+00>
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_18]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = "mhlo.sqrt"(%[[TMP_20]])
  // CHECK: %[[TMP_22:.*]] = mhlo.constant dense<1.000{{.*}}e+00>
  // CHECK: %[[TMP_23:.*]] = mhlo.add %[[TMP_22]], %[[TMP_21]]
  // CHECK: %[[TMP_24:.*]] = mhlo.divide %[[TMP_15]], %[[TMP_23]]
  // CHECK: %[[TMP_25:.*]] = mhlo.multiply %[[TMP_14]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = mhlo.add %[[TMP_13]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = "mhlo.log_plus_one"(%[[TMP_26]])
  // CHECK: %[[TMP_28:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_29:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_30:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_31:.*]] = mhlo.multiply %[[TMP_29]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = mhlo.constant dense<1.000{{.*}}e+00>
  // CHECK: %[[TMP_33:.*]] = mhlo.add %[[TMP_31]], %[[TMP_32]]
  // CHECK: %[[TMP_34:.*]] = "mhlo.sqrt"(%[[TMP_33]])
  // CHECK: %[[TMP_35:.*]] = mhlo.add %[[TMP_28]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = "mhlo.log"(%[[TMP_35]])
  // CHECK: %[[TMP_37:.*]] = "mhlo.select"(%[[TMP_12]], %[[TMP_27]], %[[TMP_36]])
  // CHECK: %[[TMP_38:.*]] = "mhlo.select"(%[[TMP_4]], %[[TMP_9]], %[[TMP_37]])
  // CHECK: %[[RES:.*]] = mhlo.multiply %[[TMP_0]], %[[TMP_38]]
  // CHECK: return %[[RES]]
  %result = "chlo.asinh"(%arg) : (tensor<f64>) -> tensor<f64>
  return %result : tensor<f64>
}

// Lower statically shaped `constant_like` to constant.
// CHECK-LABEL: @constant_like_static_shape
func @constant_like_static_shape(%arg : tensor<1x2xi64>) -> tensor<1x2xf32> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant dense<3.200000e+00> : tensor<1x2xf32>
  // CHECK: return %[[RESULT]]
  %result = "chlo.constant_like"(%arg) { value = 3.2 : f32 }
      : (tensor<1x2xi64>) -> tensor<1x2xf32>
  return %result : tensor<1x2xf32>
}

// Lower dynamically shaped `constant_like` to broadcasted constant.
// CHECK-LABEL: constant_like_dynamic_shape
// CHECK-SAME: (%[[ARG:.*]]: tensor<?x?xi64>)
func @constant_like_dynamic_shape(%arg : tensor<?x?xi64>) -> tensor<?x?xf32> {
  // CHECK: %[[CONSTANT:.*]] = mhlo.constant dense<3.200000e+00> : tensor<f32>
  // CHECK: %[[UNCASTED_SHAPE:.*]] = shape.shape_of %[[ARG]] : tensor<?x?xi64> -> tensor<?xindex>
  // CHECK: %[[SHAPE:.*]] = tensor.cast %[[UNCASTED_SHAPE]] : tensor<?xindex> to tensor<2xindex>
  // CHECK: %[[BROADCASTED_CONSTANT:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[CONSTANT]], %[[SHAPE]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK: return %[[BROADCASTED_CONSTANT]] : tensor<?x?xf32>
  %result = "chlo.constant_like"(%arg) { value = 3.2 : f32 }
      : (tensor<?x?xi64>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// CHECK-LABEL: func @conj
func @conj(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>> {
  // CHECK-SAME: ([[INPUT:%.*]]: tensor
  // CHECK-NEXT: [[R1:%.*]] = "mhlo.real"([[INPUT]])
  // CHECK-NEXT: [[R2:%.*]] = "mhlo.imag"([[INPUT]])
  // CHECK-NEXT: [[R3:%.*]] = "mhlo.negate"([[R2]])
  // CHECK-NEXT: [[R4:%.*]] = "mhlo.complex"([[R1]], [[R3]])
  %1 = "chlo.conj"(%arg0) : (tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>>
  return %1 : tensor<3xcomplex<f32>>
}

// CHECK-LABEL: @erf_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func @erf_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_0:.*]] = mhlo.constant dense<-4.000000e+00>
  // CHECK: %[[TMP_1:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_2:.*]] = "mhlo.clamp"(%[[TMP_0]], %[[ARG]], %[[TMP_1]])
  // CHECK: %[[TMP_3:.*]] = mhlo.multiply %[[TMP_2]], %[[TMP_2]]
  // CHECK: %[[TMP_4:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_5:.*]] = mhlo.multiply %[[TMP_4]], %[[TMP_3]]
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<-2.72614237E-10>
  // CHECK: %[[TMP_7:.*]] = mhlo.add %[[TMP_5]], %[[TMP_6]]
  // CHECK: %[[TMP_8:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_3]]
  // CHECK: %[[TMP_9:.*]] = mhlo.constant dense<2.77068146E-8>
  // CHECK: %[[TMP_10:.*]] = mhlo.add %[[TMP_8]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = mhlo.multiply %[[TMP_10]], %[[TMP_3]]
  // CHECK: %[[TMP_12:.*]] = mhlo.constant dense<-2.10102394E-6>
  // CHECK: %[[TMP_13:.*]] = mhlo.add %[[TMP_11]], %[[TMP_12]]
  // CHECK: %[[TMP_14:.*]] = mhlo.multiply %[[TMP_13]], %[[TMP_3]]
  // CHECK: %[[TMP_15:.*]] = mhlo.constant dense<-5.69250624E-5>
  // CHECK: %[[TMP_16:.*]] = mhlo.add %[[TMP_14]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = mhlo.multiply %[[TMP_16]], %[[TMP_3]]
  // CHECK: %[[TMP_18:.*]] = mhlo.constant dense<-7.34990637E-4>
  // CHECK: %[[TMP_19:.*]] = mhlo.add %[[TMP_17]], %[[TMP_18]]
  // CHECK: %[[TMP_20:.*]] = mhlo.multiply %[[TMP_19]], %[[TMP_3]]
  // CHECK: %[[TMP_21:.*]] = mhlo.constant dense<-2.954600e-03>
  // CHECK: %[[TMP_22:.*]] = mhlo.add %[[TMP_20]], %[[TMP_21]]
  // CHECK: %[[TMP_23:.*]] = mhlo.multiply %[[TMP_22]], %[[TMP_3]]
  // CHECK: %[[TMP_24:.*]] = mhlo.constant dense<-0.0160960332>
  // CHECK: %[[TMP_25:.*]] = mhlo.add %[[TMP_23]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_27:.*]] = mhlo.multiply %[[TMP_26]], %[[TMP_3]]
  // CHECK: %[[TMP_28:.*]] = mhlo.constant dense<-1.45660715E-5>
  // CHECK: %[[TMP_29:.*]] = mhlo.add %[[TMP_27]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = mhlo.multiply %[[TMP_29]], %[[TMP_3]]
  // CHECK: %[[TMP_31:.*]] = mhlo.constant dense<-2.13374049E-4>
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_30]], %[[TMP_31]]
  // CHECK: %[[TMP_33:.*]] = mhlo.multiply %[[TMP_32]], %[[TMP_3]]
  // CHECK: %[[TMP_34:.*]] = mhlo.constant dense<-0.00168282702>
  // CHECK: %[[TMP_35:.*]] = mhlo.add %[[TMP_33]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.multiply %[[TMP_35]], %[[TMP_3]]
  // CHECK: %[[TMP_37:.*]] = mhlo.constant dense<-0.00737332925>
  // CHECK: %[[TMP_38:.*]] = mhlo.add %[[TMP_36]], %[[TMP_37]]
  // CHECK: %[[TMP_39:.*]] = mhlo.multiply %[[TMP_38]], %[[TMP_3]]
  // CHECK: %[[TMP_40:.*]] = mhlo.constant dense<-0.0142647391>
  // CHECK: %[[TMP_41:.*]] = mhlo.add %[[TMP_39]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = mhlo.multiply %[[TMP_2]], %[[TMP_25]]
  // CHECK: %[[RESULT:.*]] = mhlo.divide %[[TMP_42]], %[[TMP_41]]
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @erf_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func @erf_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.convert"(%[[ARG]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = "mhlo.convert"(%{{.*}}) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<f16>) -> tensor<f16>
  return %1 : tensor<f16>
}

// CHECK-LABEL: @acosh
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func @acosh(%arg: tensor<f16>) -> tensor<f16> {
  // CHECK: %[[MINUSONE:.*]] = mhlo.constant dense<-1.000000e+00>
  // CHECK: %[[CMP:.*]] = "mhlo.compare"(%[[ARG]], %[[MINUSONE]]) {comparison_direction = "LT"}
  // CHECK: %[[MAX:.*]] = mhlo.constant dense<6.550400e+04>
  // CHECK: %[[SQRTMAX:.*]] = "mhlo.sqrt"(%[[MAX]])
  // CHECK: %[[OVERFLOW:.*]] = "mhlo.compare"(%[[ARG]], %[[SQRTMAX]]) {comparison_direction = "GE"}
  // CHECK: %[[LOGARG:.*]] = "mhlo.log"(%[[ARG]])
  // CHECK: %[[TWO:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[LOGTWO:.*]] = "mhlo.log"(%[[TWO]])
  // CHECK: %[[OFLRES:.*]] = mhlo.add %[[LOGARG]], %[[LOGTWO]]
  // CHECK: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[ARGPONE:.*]] = mhlo.add %[[ONE]], %[[ARG]]
  // CHECK: %[[MINUSONE2:.*]] = mhlo.constant dense<-1.000000e+00>
  // CHECK: %[[ARGMONE:.*]] = mhlo.add %[[MINUSONE2]], %[[ARG]]
  // CHECK: %[[MUL:.*]] = mhlo.multiply %[[ARGPONE]], %[[ARGMONE]]
  // CHECK: %[[SQRT:.*]] = "mhlo.sqrt"(%[[MUL]])
  // CHECK: %[[APSQRT:.*]] = mhlo.add %[[ARG]], %[[SQRT]]
  // CHECK: %[[LOGAPMUL:.*]] = "mhlo.log"(%[[APSQRT]])
  // CHECK: %[[SEL1:.*]] = "mhlo.select"(%[[OVERFLOW]], %[[OFLRES]], %[[LOGAPMUL]])
  // CHECK: %[[NAN:.*]] = mhlo.constant dense<0x7E00>
  // CHECK: %[[RESULT:.*]] = "mhlo.select"(%[[CMP]], %[[NAN]], %[[SEL1]])
  // CHECK: return %[[RESULT]]
  %1 = "chlo.acosh"(%arg) : (tensor<f16>) -> tensor<f16>
  return %1 : tensor<f16>
}
