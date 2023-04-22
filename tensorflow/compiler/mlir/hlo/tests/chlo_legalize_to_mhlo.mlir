// RUN: mlir-hlo-opt --chlo-legalize-to-hlo --split-input-file %s | FileCheck %s

// CHECK-LABEL: @asinh_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func @asinh_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  // Check for the bf16-specific max value.
  // CHECK: mhlo.constant dense<3.389{{.*}}e+38>
  %result = "chlo.asinh"(%arg) : (tensor<bf16>) -> tensor<bf16>
  return %result : tensor<bf16>
}

// ----

// CHECK-LABEL: @asinh_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func @asinh_f16(%arg : tensor<f16>) -> tensor<f16> {
  // Check for the f16-specific max value.
  // CHECK: mhlo.constant dense<6.550{{.*}}e+04>
  %result = "chlo.asinh"(%arg) : (tensor<f16>) -> tensor<f16>
  return %result : tensor<f16>
}

// ----

// CHECK-LABEL: @asinh_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func @asinh_f32(%arg : tensor<f32>) -> tensor<f32> {
  // Check for the f32-specific max value.
  // CHECK: mhlo.constant dense<3.402{{.*}}E+38>
  %result = "chlo.asinh"(%arg) : (tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}

// ----

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

// ----

// Lower statically shaped `constant_like` to constant.
// CHECK-LABEL: @constant_like_static_shape
func @constant_like_static_shape(%arg : tensor<1x2xi64>) -> tensor<1x2xf32> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant dense<3.200000e+00> : tensor<1x2xf32>
  // CHECK: return %[[RESULT]]
  %result = "chlo.constant_like"(%arg) { value = 3.2 : f32 }
      : (tensor<1x2xi64>) -> tensor<1x2xf32>
  return %result : tensor<1x2xf32>
}

// ----

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

// ----

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

// ----

// CHECK-LABEL: @erf_f64
// CHECK-SAME: %[[ARG:.*]]: tensor<f64>
func @erf_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK: %[[TMP_0:.*]] = mhlo.multiply %[[ARG]], %[[ARG]]
  // CHECK: %[[TMP_1:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_2:.*]] = mhlo.multiply %[[TMP_1]], %[[TMP_0]]
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<9.6049737398705161>
  // CHECK: %[[TMP_4:.*]] = mhlo.add %[[TMP_2]], %[[TMP_3]]
  // CHECK: %[[TMP_5:.*]] = mhlo.multiply %[[TMP_4]], %[[TMP_0]]
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<90.026019720384269>
  // CHECK: %[[TMP_7:.*]] = mhlo.add %[[TMP_5]], %[[TMP_6]]
  // CHECK: %[[TMP_8:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_0]]
  // CHECK: %[[TMP_9:.*]] = mhlo.constant dense<2232.0053459468431>
  // CHECK: %[[TMP_10:.*]] = mhlo.add %[[TMP_8]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = mhlo.multiply %[[TMP_10]], %[[TMP_0]]
  // CHECK: %[[TMP_12:.*]] = mhlo.constant dense<7003.3251411280507>
  // CHECK: %[[TMP_13:.*]] = mhlo.add %[[TMP_11]], %[[TMP_12]]
  // CHECK: %[[TMP_14:.*]] = mhlo.multiply %[[TMP_13]], %[[TMP_0]]
  // CHECK: %[[TMP_15:.*]] = mhlo.constant dense<55592.301301039493>
  // CHECK: %[[TMP_16:.*]] = mhlo.add %[[TMP_14]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = mhlo.multiply %[[ARG]], %[[TMP_16]]
  // CHECK: %[[TMP_18:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_19:.*]] = mhlo.multiply %[[TMP_18]], %[[TMP_0]]
  // CHECK: %[[TMP_20:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_21:.*]] = mhlo.add %[[TMP_19]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.multiply %[[TMP_21]], %[[TMP_0]]
  // CHECK: %[[TMP_23:.*]] = mhlo.constant dense<33.561714164750313>
  // CHECK: %[[TMP_24:.*]] = mhlo.add %[[TMP_22]], %[[TMP_23]]
  // CHECK: %[[TMP_25:.*]] = mhlo.multiply %[[TMP_24]], %[[TMP_0]]
  // CHECK: %[[TMP_26:.*]] = mhlo.constant dense<521.35794978015269>
  // CHECK: %[[TMP_27:.*]] = mhlo.add %[[TMP_25]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = mhlo.multiply %[[TMP_27]], %[[TMP_0]]
  // CHECK: %[[TMP_29:.*]] = mhlo.constant dense<4594.3238297098014>
  // CHECK: %[[TMP_30:.*]] = mhlo.add %[[TMP_28]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = mhlo.multiply %[[TMP_30]], %[[TMP_0]]
  // CHECK: %[[TMP_32:.*]] = mhlo.constant dense<22629.000061389095>
  // CHECK: %[[TMP_33:.*]] = mhlo.add %[[TMP_31]], %[[TMP_32]]
  // CHECK: %[[TMP_34:.*]] = mhlo.multiply %[[TMP_33]], %[[TMP_0]]
  // CHECK: %[[TMP_35:.*]] = mhlo.constant dense<49267.394260863592>
  // CHECK: %[[TMP_36:.*]] = mhlo.add %[[TMP_34]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.divide %[[TMP_17]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_39:.*]] = mhlo.multiply %[[ARG]], %[[ARG]]
  // CHECK: %[[TMP_40:.*]] = "mhlo.negate"(%[[TMP_39]])
  // CHECK: %[[TMP_41:.*]] = "mhlo.exponential"(%[[TMP_40]])
  // CHECK: %[[TMP_42:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_43:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_44:.*]] = mhlo.multiply %[[TMP_43]], %[[TMP_42]]
  // CHECK: %[[TMP_45:.*]] = mhlo.constant dense<2.4619698147353052E-10>
  // CHECK: %[[TMP_46:.*]] = mhlo.add %[[TMP_44]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = mhlo.multiply %[[TMP_46]], %[[TMP_42]]
  // CHECK: %[[TMP_48:.*]] = mhlo.constant dense<0.56418956483106886>
  // CHECK: %[[TMP_49:.*]] = mhlo.add %[[TMP_47]], %[[TMP_48]]
  // CHECK: %[[TMP_50:.*]] = mhlo.multiply %[[TMP_49]], %[[TMP_42]]
  // CHECK: %[[TMP_51:.*]] = mhlo.constant dense<7.4632105644226989>
  // CHECK: %[[TMP_52:.*]] = mhlo.add %[[TMP_50]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = mhlo.multiply %[[TMP_52]], %[[TMP_42]]
  // CHECK: %[[TMP_54:.*]] = mhlo.constant dense<48.637197098568137>
  // CHECK: %[[TMP_55:.*]] = mhlo.add %[[TMP_53]], %[[TMP_54]]
  // CHECK: %[[TMP_56:.*]] = mhlo.multiply %[[TMP_55]], %[[TMP_42]]
  // CHECK: %[[TMP_57:.*]] = mhlo.constant dense<196.5208329560771>
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_56]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = mhlo.multiply %[[TMP_58]], %[[TMP_42]]
  // CHECK: %[[TMP_60:.*]] = mhlo.constant dense<526.44519499547732>
  // CHECK: %[[TMP_61:.*]] = mhlo.add %[[TMP_59]], %[[TMP_60]]
  // CHECK: %[[TMP_62:.*]] = mhlo.multiply %[[TMP_61]], %[[TMP_42]]
  // CHECK: %[[TMP_63:.*]] = mhlo.constant dense<934.52852717195765>
  // CHECK: %[[TMP_64:.*]] = mhlo.add %[[TMP_62]], %[[TMP_63]]
  // CHECK: %[[TMP_65:.*]] = mhlo.multiply %[[TMP_64]], %[[TMP_42]]
  // CHECK: %[[TMP_66:.*]] = mhlo.constant dense<1027.5518868951572>
  // CHECK: %[[TMP_67:.*]] = mhlo.add %[[TMP_65]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = mhlo.multiply %[[TMP_67]], %[[TMP_42]]
  // CHECK: %[[TMP_69:.*]] = mhlo.constant dense<557.53533536939938>
  // CHECK: %[[TMP_70:.*]] = mhlo.add %[[TMP_68]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_70]]
  // CHECK: %[[TMP_72:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_73:.*]] = mhlo.multiply %[[TMP_72]], %[[TMP_42]]
  // CHECK: %[[TMP_74:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_75:.*]] = mhlo.add %[[TMP_73]], %[[TMP_74]]
  // CHECK: %[[TMP_76:.*]] = mhlo.multiply %[[TMP_75]], %[[TMP_42]]
  // CHECK: %[[TMP_77:.*]] = mhlo.constant dense<13.228195115474499>
  // CHECK: %[[TMP_78:.*]] = mhlo.add %[[TMP_76]], %[[TMP_77]]
  // CHECK: %[[TMP_79:.*]] = mhlo.multiply %[[TMP_78]], %[[TMP_42]]
  // CHECK: %[[TMP_80:.*]] = mhlo.constant dense<86.707214088598973>
  // CHECK: %[[TMP_81:.*]] = mhlo.add %[[TMP_79]], %[[TMP_80]]
  // CHECK: %[[TMP_82:.*]] = mhlo.multiply %[[TMP_81]], %[[TMP_42]]
  // CHECK: %[[TMP_83:.*]] = mhlo.constant dense<354.93777888781989>
  // CHECK: %[[TMP_84:.*]] = mhlo.add %[[TMP_82]], %[[TMP_83]]
  // CHECK: %[[TMP_85:.*]] = mhlo.multiply %[[TMP_84]], %[[TMP_42]]
  // CHECK: %[[TMP_86:.*]] = mhlo.constant dense<975.70850174320549>
  // CHECK: %[[TMP_87:.*]] = mhlo.add %[[TMP_85]], %[[TMP_86]]
  // CHECK: %[[TMP_88:.*]] = mhlo.multiply %[[TMP_87]], %[[TMP_42]]
  // CHECK: %[[TMP_89:.*]] = mhlo.constant dense<1823.9091668790973>
  // CHECK: %[[TMP_90:.*]] = mhlo.add %[[TMP_88]], %[[TMP_89]]
  // CHECK: %[[TMP_91:.*]] = mhlo.multiply %[[TMP_90]], %[[TMP_42]]
  // CHECK: %[[TMP_92:.*]] = mhlo.constant dense<2246.3376081871097>
  // CHECK: %[[TMP_93:.*]] = mhlo.add %[[TMP_91]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = mhlo.multiply %[[TMP_93]], %[[TMP_42]]
  // CHECK: %[[TMP_95:.*]] = mhlo.constant dense<1656.6630919416134>
  // CHECK: %[[TMP_96:.*]] = mhlo.add %[[TMP_94]], %[[TMP_95]]
  // CHECK: %[[TMP_97:.*]] = mhlo.multiply %[[TMP_96]], %[[TMP_42]]
  // CHECK: %[[TMP_98:.*]] = mhlo.constant dense<557.53534081772773>
  // CHECK: %[[TMP_99:.*]] = mhlo.add %[[TMP_97]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.divide %[[TMP_71]], %[[TMP_99]]
  // CHECK: %[[TMP_101:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_102:.*]] = mhlo.multiply %[[TMP_101]], %[[TMP_42]]
  // CHECK: %[[TMP_103:.*]] = mhlo.constant dense<0.56418958354775506>
  // CHECK: %[[TMP_104:.*]] = mhlo.add %[[TMP_102]], %[[TMP_103]]
  // CHECK: %[[TMP_105:.*]] = mhlo.multiply %[[TMP_104]], %[[TMP_42]]
  // CHECK: %[[TMP_106:.*]] = mhlo.constant dense<1.275366707599781>
  // CHECK: %[[TMP_107:.*]] = mhlo.add %[[TMP_105]], %[[TMP_106]]
  // CHECK: %[[TMP_108:.*]] = mhlo.multiply %[[TMP_107]], %[[TMP_42]]
  // CHECK: %[[TMP_109:.*]] = mhlo.constant dense<5.0190504225118051>
  // CHECK: %[[TMP_110:.*]] = mhlo.add %[[TMP_108]], %[[TMP_109]]
  // CHECK: %[[TMP_111:.*]] = mhlo.multiply %[[TMP_110]], %[[TMP_42]]
  // CHECK: %[[TMP_112:.*]] = mhlo.constant dense<6.160210979930536>
  // CHECK: %[[TMP_113:.*]] = mhlo.add %[[TMP_111]], %[[TMP_112]]
  // CHECK: %[[TMP_114:.*]] = mhlo.multiply %[[TMP_113]], %[[TMP_42]]
  // CHECK: %[[TMP_115:.*]] = mhlo.constant dense<7.4097426995044895>
  // CHECK: %[[TMP_116:.*]] = mhlo.add %[[TMP_114]], %[[TMP_115]]
  // CHECK: %[[TMP_117:.*]] = mhlo.multiply %[[TMP_116]], %[[TMP_42]]
  // CHECK: %[[TMP_118:.*]] = mhlo.constant dense<2.9788666537210022>
  // CHECK: %[[TMP_119:.*]] = mhlo.add %[[TMP_117]], %[[TMP_118]]
  // CHECK: %[[TMP_120:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_119]]
  // CHECK: %[[TMP_121:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_122:.*]] = mhlo.multiply %[[TMP_121]], %[[TMP_42]]
  // CHECK: %[[TMP_123:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_124:.*]] = mhlo.add %[[TMP_122]], %[[TMP_123]]
  // CHECK: %[[TMP_125:.*]] = mhlo.multiply %[[TMP_124]], %[[TMP_42]]
  // CHECK: %[[TMP_126:.*]] = mhlo.constant dense<2.2605286322011726>
  // CHECK: %[[TMP_127:.*]] = mhlo.add %[[TMP_125]], %[[TMP_126]]
  // CHECK: %[[TMP_128:.*]] = mhlo.multiply %[[TMP_127]], %[[TMP_42]]
  // CHECK: %[[TMP_129:.*]] = mhlo.constant dense<9.3960352493800147>
  // CHECK: %[[TMP_130:.*]] = mhlo.add %[[TMP_128]], %[[TMP_129]]
  // CHECK: %[[TMP_131:.*]] = mhlo.multiply %[[TMP_130]], %[[TMP_42]]
  // CHECK: %[[TMP_132:.*]] = mhlo.constant dense<12.048953980809666>
  // CHECK: %[[TMP_133:.*]] = mhlo.add %[[TMP_131]], %[[TMP_132]]
  // CHECK: %[[TMP_134:.*]] = mhlo.multiply %[[TMP_133]], %[[TMP_42]]
  // CHECK: %[[TMP_135:.*]] = mhlo.constant dense<17.081445074756591>
  // CHECK: %[[TMP_136:.*]] = mhlo.add %[[TMP_134]], %[[TMP_135]]
  // CHECK: %[[TMP_137:.*]] = mhlo.multiply %[[TMP_136]], %[[TMP_42]]
  // CHECK: %[[TMP_138:.*]] = mhlo.constant dense<9.6089680906328585>
  // CHECK: %[[TMP_139:.*]] = mhlo.add %[[TMP_137]], %[[TMP_138]]
  // CHECK: %[[TMP_140:.*]] = mhlo.multiply %[[TMP_139]], %[[TMP_42]]
  // CHECK: %[[TMP_141:.*]] = mhlo.constant dense<3.3690764510008151>
  // CHECK: %[[TMP_142:.*]] = mhlo.add %[[TMP_140]], %[[TMP_141]]
  // CHECK: %[[TMP_143:.*]] = mhlo.divide %[[TMP_120]], %[[TMP_142]]
  // CHECK: %[[TMP_144:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_145:.*]] = "mhlo.compare"(%[[TMP_42]], %[[TMP_144]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_146:.*]] = "mhlo.select"(%[[TMP_145]], %[[TMP_100]], %[[TMP_143]])
  // CHECK: %[[TMP_147:.*]] = mhlo.constant dense<-709.78271289338397>
  // CHECK: %[[TMP_148:.*]] = "mhlo.compare"(%[[TMP_40]], %[[TMP_147]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_149:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_150:.*]] = "mhlo.select"(%[[TMP_148]], %[[TMP_149]], %[[TMP_146]])
  // CHECK: %[[TMP_152:.*]] = "mhlo.compare"(%[[ARG]], %[[TMP_149]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_153:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_154:.*]] = mhlo.subtract %[[TMP_153]], %[[TMP_150]]
  // CHECK: %[[TMP_155:.*]] = "mhlo.select"(%[[TMP_152]], %[[TMP_154]], %[[TMP_150]])
  // CHECK: %[[TMP_156:.*]] = mhlo.subtract %[[TMP_38]], %[[TMP_155]]
  // CHECK: %[[TMP_157:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_159:.*]] = "mhlo.compare"(%[[TMP_157]], %[[TMP_38]]) {comparison_direction = "LT"}
  // CHECK: %[[RESULT:.*]] = "mhlo.select"(%[[TMP_159]], %[[TMP_37]], %[[TMP_156]])
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<f64>) -> tensor<f64>
  return %1 : tensor<f64>
}

// ----

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

// ----

// CHECK-LABEL: @erf_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func @erf_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.convert"(%[[ARG]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = "mhlo.convert"(%{{.*}}) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<f16>) -> tensor<f16>
  return %1 : tensor<f16>
}

// ----

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

// ----

// CHECK-LABEL: @erfc_f64
// CHECK-SAME: %[[ARG:.*]]: tensor<f64>
func @erfc_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK-NEXT: %[[TMP_0:.*]] = mhlo.multiply %[[ARG]], %[[ARG]]
  // CHECK-NEXT: %[[TMP_1:.*]] = "mhlo.negate"(%[[TMP_0]])
  // CHECK-NEXT: %[[TMP_2:.*]] = "mhlo.exponential"(%[[TMP_1]])
  // CHECK-NEXT: %[[TMP_3:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK-NEXT: %[[TMP_4:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[TMP_5:.*]] = mhlo.multiply %[[TMP_4]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_6:.*]] = mhlo.constant dense<2.4619698147353052E-10>
  // CHECK-NEXT: %[[TMP_7:.*]] = mhlo.add %[[TMP_5]], %[[TMP_6]]
  // CHECK-NEXT: %[[TMP_8:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_9:.*]] = mhlo.constant dense<0.56418956483106886>
  // CHECK-NEXT: %[[TMP_10:.*]] = mhlo.add %[[TMP_8]], %[[TMP_9]]
  // CHECK-NEXT: %[[TMP_11:.*]] = mhlo.multiply %[[TMP_10]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_12:.*]] = mhlo.constant dense<7.4632105644226989>
  // CHECK-NEXT: %[[TMP_13:.*]] = mhlo.add %[[TMP_11]], %[[TMP_12]]
  // CHECK-NEXT: %[[TMP_14:.*]] = mhlo.multiply %[[TMP_13]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_15:.*]] = mhlo.constant dense<48.637197098568137>
  // CHECK-NEXT: %[[TMP_16:.*]] = mhlo.add %[[TMP_14]], %[[TMP_15]]
  // CHECK-NEXT: %[[TMP_17:.*]] = mhlo.multiply %[[TMP_16]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_18:.*]] = mhlo.constant dense<196.5208329560771>
  // CHECK-NEXT: %[[TMP_19:.*]] = mhlo.add %[[TMP_17]], %[[TMP_18]]
  // CHECK-NEXT: %[[TMP_20:.*]] = mhlo.multiply %[[TMP_19]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_21:.*]] = mhlo.constant dense<526.44519499547732>
  // CHECK-NEXT: %[[TMP_22:.*]] = mhlo.add %[[TMP_20]], %[[TMP_21]]
  // CHECK-NEXT: %[[TMP_23:.*]] = mhlo.multiply %[[TMP_22]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_24:.*]] = mhlo.constant dense<934.52852717195765>
  // CHECK-NEXT: %[[TMP_25:.*]] = mhlo.add %[[TMP_23]], %[[TMP_24]]
  // CHECK-NEXT: %[[TMP_26:.*]] = mhlo.multiply %[[TMP_25]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_27:.*]] = mhlo.constant dense<1027.5518868951572>
  // CHECK-NEXT: %[[TMP_28:.*]] = mhlo.add %[[TMP_26]], %[[TMP_27]]
  // CHECK-NEXT: %[[TMP_29:.*]] = mhlo.multiply %[[TMP_28]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_30:.*]] = mhlo.constant dense<557.53533536939938>
  // CHECK-NEXT: %[[TMP_31:.*]] = mhlo.add %[[TMP_29]], %[[TMP_30]]
  // CHECK-NEXT: %[[TMP_32:.*]] = mhlo.multiply %[[TMP_2]], %[[TMP_31]]
  // CHECK-NEXT: %[[TMP_33:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[TMP_34:.*]] = mhlo.multiply %[[TMP_33]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_35:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_36:.*]] = mhlo.add %[[TMP_34]], %[[TMP_35]]
  // CHECK-NEXT: %[[TMP_37:.*]] = mhlo.multiply %[[TMP_36]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_38:.*]] = mhlo.constant dense<13.228195115474499>
  // CHECK-NEXT: %[[TMP_39:.*]] = mhlo.add %[[TMP_37]], %[[TMP_38]]
  // CHECK-NEXT: %[[TMP_40:.*]] = mhlo.multiply %[[TMP_39]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_41:.*]] = mhlo.constant dense<86.707214088598973>
  // CHECK-NEXT: %[[TMP_42:.*]] = mhlo.add %[[TMP_40]], %[[TMP_41]]
  // CHECK-NEXT: %[[TMP_43:.*]] = mhlo.multiply %[[TMP_42]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_44:.*]] = mhlo.constant dense<354.93777888781989>
  // CHECK-NEXT: %[[TMP_45:.*]] = mhlo.add %[[TMP_43]], %[[TMP_44]]
  // CHECK-NEXT: %[[TMP_46:.*]] = mhlo.multiply %[[TMP_45]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_47:.*]] = mhlo.constant dense<975.70850174320549>
  // CHECK-NEXT: %[[TMP_48:.*]] = mhlo.add %[[TMP_46]], %[[TMP_47]]
  // CHECK-NEXT: %[[TMP_49:.*]] = mhlo.multiply %[[TMP_48]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_50:.*]] = mhlo.constant dense<1823.9091668790973>
  // CHECK-NEXT: %[[TMP_51:.*]] = mhlo.add %[[TMP_49]], %[[TMP_50]]
  // CHECK-NEXT: %[[TMP_52:.*]] = mhlo.multiply %[[TMP_51]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_53:.*]] = mhlo.constant dense<2246.3376081871097>
  // CHECK-NEXT: %[[TMP_54:.*]] = mhlo.add %[[TMP_52]], %[[TMP_53]]
  // CHECK-NEXT: %[[TMP_55:.*]] = mhlo.multiply %[[TMP_54]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_56:.*]] = mhlo.constant dense<1656.6630919416134>
  // CHECK-NEXT: %[[TMP_57:.*]] = mhlo.add %[[TMP_55]], %[[TMP_56]]
  // CHECK-NEXT: %[[TMP_58:.*]] = mhlo.multiply %[[TMP_57]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_59:.*]] = mhlo.constant dense<557.53534081772773>
  // CHECK-NEXT: %[[TMP_60:.*]] = mhlo.add %[[TMP_58]], %[[TMP_59]]
  // CHECK-NEXT: %[[TMP_61:.*]] = mhlo.divide %[[TMP_32]], %[[TMP_60]]
  // CHECK-NEXT: %[[TMP_62:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[TMP_63:.*]] = mhlo.multiply %[[TMP_62]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_64:.*]] = mhlo.constant dense<0.56418958354775506>
  // CHECK-NEXT: %[[TMP_65:.*]] = mhlo.add %[[TMP_63]], %[[TMP_64]]
  // CHECK-NEXT: %[[TMP_66:.*]] = mhlo.multiply %[[TMP_65]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_67:.*]] = mhlo.constant dense<1.275366707599781>
  // CHECK-NEXT: %[[TMP_68:.*]] = mhlo.add %[[TMP_66]], %[[TMP_67]]
  // CHECK-NEXT: %[[TMP_69:.*]] = mhlo.multiply %[[TMP_68]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_70:.*]] = mhlo.constant dense<5.0190504225118051>
  // CHECK-NEXT: %[[TMP_71:.*]] = mhlo.add %[[TMP_69]], %[[TMP_70]]
  // CHECK-NEXT: %[[TMP_72:.*]] = mhlo.multiply %[[TMP_71]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_73:.*]] = mhlo.constant dense<6.160210979930536>
  // CHECK-NEXT: %[[TMP_74:.*]] = mhlo.add %[[TMP_72]], %[[TMP_73]]
  // CHECK-NEXT: %[[TMP_75:.*]] = mhlo.multiply %[[TMP_74]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_76:.*]] = mhlo.constant dense<7.4097426995044895>
  // CHECK-NEXT: %[[TMP_77:.*]] = mhlo.add %[[TMP_75]], %[[TMP_76]]
  // CHECK-NEXT: %[[TMP_78:.*]] = mhlo.multiply %[[TMP_77]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_79:.*]] = mhlo.constant dense<2.9788666537210022>
  // CHECK-NEXT: %[[TMP_80:.*]] = mhlo.add %[[TMP_78]], %[[TMP_79]]
  // CHECK-NEXT: %[[TMP_81:.*]] = mhlo.multiply %[[TMP_2]], %[[TMP_80]]
  // CHECK-NEXT: %[[TMP_82:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[TMP_83:.*]] = mhlo.multiply %[[TMP_82]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_84:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_85:.*]] = mhlo.add %[[TMP_83]], %[[TMP_84]]
  // CHECK-NEXT: %[[TMP_86:.*]] = mhlo.multiply %[[TMP_85]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_87:.*]] = mhlo.constant dense<2.2605286322011726>
  // CHECK-NEXT: %[[TMP_88:.*]] = mhlo.add %[[TMP_86]], %[[TMP_87]]
  // CHECK-NEXT: %[[TMP_89:.*]] = mhlo.multiply %[[TMP_88]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_90:.*]] = mhlo.constant dense<9.3960352493800147>
  // CHECK-NEXT: %[[TMP_91:.*]] = mhlo.add %[[TMP_89]], %[[TMP_90]]
  // CHECK-NEXT: %[[TMP_92:.*]] = mhlo.multiply %[[TMP_91]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_93:.*]] = mhlo.constant dense<12.048953980809666>
  // CHECK-NEXT: %[[TMP_94:.*]] = mhlo.add %[[TMP_92]], %[[TMP_93]]
  // CHECK-NEXT: %[[TMP_95:.*]] = mhlo.multiply %[[TMP_94]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_96:.*]] = mhlo.constant dense<17.081445074756591>
  // CHECK-NEXT: %[[TMP_97:.*]] = mhlo.add %[[TMP_95]], %[[TMP_96]]
  // CHECK-NEXT: %[[TMP_98:.*]] = mhlo.multiply %[[TMP_97]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_99:.*]] = mhlo.constant dense<9.6089680906328585>
  // CHECK-NEXT: %[[TMP_100:.*]] = mhlo.add %[[TMP_98]], %[[TMP_99]]
  // CHECK-NEXT: %[[TMP_101:.*]] = mhlo.multiply %[[TMP_100]], %[[TMP_3]]
  // CHECK-NEXT: %[[TMP_102:.*]] = mhlo.constant dense<3.3690764510008151>
  // CHECK-NEXT: %[[TMP_103:.*]] = mhlo.add %[[TMP_101]], %[[TMP_102]]
  // CHECK-NEXT: %[[TMP_104:.*]] = mhlo.divide %[[TMP_81]], %[[TMP_103]]
  // CHECK-NEXT: %[[TMP_105:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK-NEXT: %[[TMP_106:.*]] = "mhlo.compare"(%[[TMP_3]], %[[TMP_105]]) {comparison_direction = "LT"}
  // CHECK-NEXT: %[[TMP_107:.*]] = "mhlo.select"(%[[TMP_106]], %[[TMP_61]], %[[TMP_104]])
  // CHECK-NEXT: %[[TMP_108:.*]] = mhlo.constant dense<-709.78271289338397>
  // CHECK-NEXT: %[[TMP_109:.*]] = "mhlo.compare"(%[[TMP_1]], %[[TMP_108]]) {comparison_direction = "LT"}
  // CHECK-NEXT: %[[TMP_110:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[TMP_111:.*]] = "mhlo.select"(%[[TMP_109]], %[[TMP_110]], %[[TMP_107]])
  // CHECK-NEXT: %[[TMP_113:.*]] = "mhlo.compare"(%[[ARG]], %[[TMP_110]]) {comparison_direction = "LT"}
  // CHECK-NEXT: %[[TMP_114:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK-NEXT: %[[TMP_115:.*]] = mhlo.subtract %[[TMP_114]], %[[TMP_111]]
  // CHECK-NEXT: %[[TMP_116:.*]] = "mhlo.select"(%[[TMP_113]], %[[TMP_115]], %[[TMP_111]])
  // CHECK-NEXT: %[[TMP_117:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_118:.*]] = mhlo.multiply %[[ARG]], %[[ARG]]
  // CHECK-NEXT: %[[TMP_119:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[TMP_120:.*]] = mhlo.multiply %[[TMP_119]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_121:.*]] = mhlo.constant dense<9.6049737398705161>
  // CHECK-NEXT: %[[TMP_122:.*]] = mhlo.add %[[TMP_120]], %[[TMP_121]]
  // CHECK-NEXT: %[[TMP_123:.*]] = mhlo.multiply %[[TMP_122]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_124:.*]] = mhlo.constant dense<90.026019720384269>
  // CHECK-NEXT: %[[TMP_125:.*]] = mhlo.add %[[TMP_123]], %[[TMP_124]]
  // CHECK-NEXT: %[[TMP_126:.*]] = mhlo.multiply %[[TMP_125]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_127:.*]] = mhlo.constant dense<2232.0053459468431>
  // CHECK-NEXT: %[[TMP_128:.*]] = mhlo.add %[[TMP_126]], %[[TMP_127]]
  // CHECK-NEXT: %[[TMP_129:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_130:.*]] = mhlo.constant dense<7003.3251411280507>
  // CHECK-NEXT: %[[TMP_131:.*]] = mhlo.add %[[TMP_129]], %[[TMP_130]]
  // CHECK-NEXT: %[[TMP_132:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_133:.*]] = mhlo.constant dense<55592.301301039493>
  // CHECK-NEXT: %[[TMP_134:.*]] = mhlo.add %[[TMP_132]], %[[TMP_133]]
  // CHECK-NEXT: %[[TMP_135:.*]] = mhlo.multiply %[[ARG]], %[[TMP_134]]
  // CHECK-NEXT: %[[TMP_136:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[TMP_137:.*]] = mhlo.multiply %[[TMP_136]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_138:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_139:.*]] = mhlo.add %[[TMP_137]], %[[TMP_138]]
  // CHECK-NEXT: %[[TMP_140:.*]] = mhlo.multiply %[[TMP_139]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_141:.*]] = mhlo.constant dense<33.561714164750313>
  // CHECK-NEXT: %[[TMP_142:.*]] = mhlo.add %[[TMP_140]], %[[TMP_141]]
  // CHECK-NEXT: %[[TMP_143:.*]] = mhlo.multiply %[[TMP_142]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_144:.*]] = mhlo.constant dense<521.35794978015269>
  // CHECK-NEXT: %[[TMP_145:.*]] = mhlo.add %[[TMP_143]], %[[TMP_144]]
  // CHECK-NEXT: %[[TMP_146:.*]] = mhlo.multiply %[[TMP_145]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_147:.*]] = mhlo.constant dense<4594.3238297098014>
  // CHECK-NEXT: %[[TMP_148:.*]] = mhlo.add %[[TMP_146]], %[[TMP_147]]
  // CHECK-NEXT: %[[TMP_149:.*]] = mhlo.multiply %[[TMP_148]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_150:.*]] = mhlo.constant dense<22629.000061389095>
  // CHECK-NEXT: %[[TMP_151:.*]] = mhlo.add %[[TMP_149]], %[[TMP_150]]
  // CHECK-NEXT: %[[TMP_152:.*]] = mhlo.multiply %[[TMP_151]], %[[TMP_118]]
  // CHECK-NEXT: %[[TMP_153:.*]] = mhlo.constant dense<49267.394260863592>
  // CHECK-NEXT: %[[TMP_154:.*]] = mhlo.add %[[TMP_152]], %[[TMP_153]]
  // CHECK-NEXT: %[[TMP_155:.*]] = mhlo.divide %[[TMP_135]], %[[TMP_154]]
  // CHECK-NEXT: %[[TMP_156:.*]] = mhlo.subtract %[[TMP_117]], %[[TMP_155]]
  // CHECK-NEXT: %[[TMP_157:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK-NEXT: %[[TMP_159:.*]] = "mhlo.compare"(%[[TMP_157]], %[[TMP_117]]) {comparison_direction = "LT"}
  // CHECK-NEXT: %[[RESULT:.*]] = "mhlo.select"(%[[TMP_159]], %[[TMP_156]], %[[TMP_116]])
  // CHECK-NEXT: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<f64>) -> tensor<f64>
  return %1 : tensor<f64>
}

// ----

// CHECK-LABEL: @erfc_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func @erfc_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_0:.*]] = mhlo.multiply %[[ARG]], %[[ARG]]
  // CHECK: %[[TMP_1:.*]] = "mhlo.negate"(%[[TMP_0]])
  // CHECK: %[[TMP_2:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_4:.*]] = mhlo.divide %[[TMP_3]], %[[TMP_0]]
  // CHECK: %[[TMP_5:.*]] = "mhlo.exponential"(%[[TMP_1]])
  // CHECK: %[[TMP_7:.*]] = mhlo.divide %[[TMP_3]], %[[TMP_2]]
  // CHECK: %[[TMP_8:.*]] = mhlo.multiply %[[TMP_5]], %[[TMP_7]]
  // CHECK: %[[TMP_9:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_10:.*]] = "mhlo.compare"(%[[TMP_2]], %[[TMP_9]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_11:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_12:.*]] = mhlo.multiply %[[TMP_11]], %[[TMP_4]]
  // CHECK: %[[TMP_13:.*]] = mhlo.constant dense<2.326820e-02>
  // CHECK: %[[TMP_14:.*]] = mhlo.add %[[TMP_12]], %[[TMP_13]]
  // CHECK: %[[TMP_15:.*]] = mhlo.multiply %[[TMP_14]], %[[TMP_4]]
  // CHECK: %[[TMP_16:.*]] = mhlo.constant dense<-0.138703942>
  // CHECK: %[[TMP_17:.*]] = mhlo.add %[[TMP_15]], %[[TMP_16]]
  // CHECK: %[[TMP_18:.*]] = mhlo.multiply %[[TMP_17]], %[[TMP_4]]
  // CHECK: %[[TMP_19:.*]] = mhlo.constant dense<0.368742466>
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_18]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.multiply %[[TMP_20]], %[[TMP_4]]
  // CHECK: %[[TMP_22:.*]] = mhlo.constant dense<-0.582473278>
  // CHECK: %[[TMP_23:.*]] = mhlo.add %[[TMP_21]], %[[TMP_22]]
  // CHECK: %[[TMP_24:.*]] = mhlo.multiply %[[TMP_23]], %[[TMP_4]]
  // CHECK: %[[TMP_25:.*]] = mhlo.constant dense<0.621000468>
  // CHECK: %[[TMP_26:.*]] = mhlo.add %[[TMP_24]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.multiply %[[TMP_26]], %[[TMP_4]]
  // CHECK: %[[TMP_28:.*]] = mhlo.constant dense<-0.494451523>
  // CHECK: %[[TMP_29:.*]] = mhlo.add %[[TMP_27]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = mhlo.multiply %[[TMP_29]], %[[TMP_4]]
  // CHECK: %[[TMP_31:.*]] = mhlo.constant dense<3.404880e-01>
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_30]], %[[TMP_31]]
  // CHECK: %[[TMP_33:.*]] = mhlo.multiply %[[TMP_32]], %[[TMP_4]]
  // CHECK: %[[TMP_34:.*]] = mhlo.constant dense<-0.274112701>
  // CHECK: %[[TMP_35:.*]] = mhlo.add %[[TMP_33]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.multiply %[[TMP_35]], %[[TMP_4]]
  // CHECK: %[[TMP_37:.*]] = mhlo.constant dense<0.563825965>
  // CHECK: %[[TMP_38:.*]] = mhlo.add %[[TMP_36]], %[[TMP_37]]
  // CHECK: %[[TMP_39:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_40:.*]] = mhlo.multiply %[[TMP_39]], %[[TMP_4]]
  // CHECK: %[[TMP_41:.*]] = mhlo.constant dense<-10.477664>
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_40]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = mhlo.multiply %[[TMP_42]], %[[TMP_4]]
  // CHECK: %[[TMP_44:.*]] = mhlo.constant dense<1.297720e+01>
  // CHECK: %[[TMP_45:.*]] = mhlo.add %[[TMP_43]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.multiply %[[TMP_45]], %[[TMP_4]]
  // CHECK: %[[TMP_47:.*]] = mhlo.constant dense<-7.49551868>
  // CHECK: %[[TMP_48:.*]] = mhlo.add %[[TMP_46]], %[[TMP_47]]
  // CHECK: %[[TMP_49:.*]] = mhlo.multiply %[[TMP_48]], %[[TMP_4]]
  // CHECK: %[[TMP_50:.*]] = mhlo.constant dense<2.92101908>
  // CHECK: %[[TMP_51:.*]] = mhlo.add %[[TMP_49]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.multiply %[[TMP_51]], %[[TMP_4]]
  // CHECK: %[[TMP_53:.*]] = mhlo.constant dense<-1.01526523>
  // CHECK: %[[TMP_54:.*]] = mhlo.add %[[TMP_52]], %[[TMP_53]]
  // CHECK: %[[TMP_55:.*]] = mhlo.multiply %[[TMP_54]], %[[TMP_4]]
  // CHECK: %[[TMP_56:.*]] = mhlo.constant dense<0.42184633>
  // CHECK: %[[TMP_57:.*]] = mhlo.add %[[TMP_55]], %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = mhlo.multiply %[[TMP_57]], %[[TMP_4]]
  // CHECK: %[[TMP_59:.*]] = mhlo.constant dense<-0.282076746>
  // CHECK: %[[TMP_60:.*]] = mhlo.add %[[TMP_58]], %[[TMP_59]]
  // CHECK: %[[TMP_61:.*]] = mhlo.multiply %[[TMP_60]], %[[TMP_4]]
  // CHECK: %[[TMP_62:.*]] = mhlo.constant dense<0.564189494>
  // CHECK: %[[TMP_63:.*]] = mhlo.add %[[TMP_61]], %[[TMP_62]]
  // CHECK: %[[TMP_64:.*]] = "mhlo.select"(%[[TMP_10]], %[[TMP_38]], %[[TMP_63]])
  // CHECK: %[[TMP_65:.*]] = mhlo.multiply %[[TMP_8]], %[[TMP_64]]
  // CHECK: %[[TMP_66:.*]] = mhlo.constant dense<-88.7228394>
  // CHECK: %[[TMP_67:.*]] = "mhlo.compare"(%[[TMP_1]], %[[TMP_66]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_68:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_69:.*]] = "mhlo.select"(%[[TMP_67]], %[[TMP_68]], %[[TMP_65]])
  // CHECK: %[[TMP_71:.*]] = "mhlo.compare"(%[[ARG]], %[[TMP_68]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_73:.*]] = mhlo.subtract %[[TMP_9]], %[[TMP_69]]
  // CHECK: %[[TMP_74:.*]] = "mhlo.select"(%[[TMP_71]], %[[TMP_73]], %[[TMP_69]])
  // CHECK: %[[TMP_75:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_76:.*]] = mhlo.multiply %[[ARG]], %[[ARG]]
  // CHECK: %[[TMP_77:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_78:.*]] = mhlo.multiply %[[TMP_77]], %[[TMP_76]]
  // CHECK: %[[TMP_79:.*]] = mhlo.constant dense<7.85386146E-5>
  // CHECK: %[[TMP_80:.*]] = mhlo.add %[[TMP_78]], %[[TMP_79]]
  // CHECK: %[[TMP_81:.*]] = mhlo.multiply %[[TMP_80]], %[[TMP_76]]
  // CHECK: %[[TMP_82:.*]] = mhlo.constant dense<-8.0101937E-4>
  // CHECK: %[[TMP_83:.*]] = mhlo.add %[[TMP_81]], %[[TMP_82]]
  // CHECK: %[[TMP_84:.*]] = mhlo.multiply %[[TMP_83]], %[[TMP_76]]
  // CHECK: %[[TMP_85:.*]] = mhlo.constant dense<0.00518832775>
  // CHECK: %[[TMP_86:.*]] = mhlo.add %[[TMP_84]], %[[TMP_85]]
  // CHECK: %[[TMP_87:.*]] = mhlo.multiply %[[TMP_86]], %[[TMP_76]]
  // CHECK: %[[TMP_88:.*]] = mhlo.constant dense<-0.0268538129>
  // CHECK: %[[TMP_89:.*]] = mhlo.add %[[TMP_87]], %[[TMP_88]]
  // CHECK: %[[TMP_90:.*]] = mhlo.multiply %[[TMP_89]], %[[TMP_76]]
  // CHECK: %[[TMP_91:.*]] = mhlo.constant dense<0.112835854>
  // CHECK: %[[TMP_92:.*]] = mhlo.add %[[TMP_90]], %[[TMP_91]]
  // CHECK: %[[TMP_93:.*]] = mhlo.multiply %[[TMP_92]], %[[TMP_76]]
  // CHECK: %[[TMP_94:.*]] = mhlo.constant dense<-0.37612626>
  // CHECK: %[[TMP_95:.*]] = mhlo.add %[[TMP_93]], %[[TMP_94]]
  // CHECK: %[[TMP_96:.*]] = mhlo.multiply %[[TMP_95]], %[[TMP_76]]
  // CHECK: %[[TMP_97:.*]] = mhlo.constant dense<1.12837911>
  // CHECK: %[[TMP_98:.*]] = mhlo.add %[[TMP_96]], %[[TMP_97]]
  // CHECK: %[[TMP_99:.*]] = mhlo.multiply %[[ARG]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.subtract %[[TMP_75]], %[[TMP_99]]
  // CHECK: %[[TMP_101:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_103:.*]] = "mhlo.compare"(%[[TMP_101]], %[[TMP_75]]) {comparison_direction = "LT"}
  // CHECK: %[[RESULT:.*]] = "mhlo.select"(%[[TMP_103]], %[[TMP_100]], %[[TMP_74]])
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// ----

// CHECK-LABEL: @erfc_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func @erfc_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.convert"(%[[ARG]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = "mhlo.convert"(%{{.*}}) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<f16>) -> tensor<f16>
  return %1 : tensor<f16>
}

// ----

// CHECK-LABEL: @is_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func @is_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  // CHECK: %[[ABS:.*]] = "mhlo.abs"(%arg0) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[POS_INF:.*]] = mhlo.constant dense<0x7F800000> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = "mhlo.compare"(%[[ABS]], %[[POS_INF]]) {comparison_direction = "EQ"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: return %[[RESULT]] : tensor<i1>
  %1 = chlo.is_inf %arg : tensor<f32> -> tensor<i1>
  return %1 : tensor<i1>
}

// ----

// CHECK-LABEL: @is_pos_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func @is_pos_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  // CHECK: %[[POS_INF:.*]] = mhlo.constant dense<0x7F800000> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = "mhlo.compare"(%[[ARG]], %[[POS_INF]]) {comparison_direction = "EQ"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: return %[[RESULT]] : tensor<i1>
  %1 = chlo.is_pos_inf %arg : tensor<f32> -> tensor<i1>
  return %1 : tensor<i1>
}

// ----

// CHECK-LABEL: @is_neg_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func @is_neg_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  // CHECK: %[[NEG_INF:.*]] = mhlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = "mhlo.compare"(%[[ARG]], %[[NEG_INF]]) {comparison_direction = "EQ"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: return %[[RESULT]] : tensor<i1>
  %1 = chlo.is_neg_inf %arg : tensor<f32> -> tensor<i1>
  return %1 : tensor<i1>
}

// ----

// CHECK-LABEL: @lgamma_f64
// CHECK-SAME: (%[[ARG:.*]]: tensor<f64>)
func @lgamma_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK: %[[TMP_1:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_9:.*]] = "mhlo.compare"(%[[ARG]], %[[TMP_1]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_10:.*]] = "mhlo.negate"(%[[ARG]])
  // CHECK: %[[TMP_2:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_11:.*]] = mhlo.subtract %[[ARG]], %[[TMP_2]]
  // CHECK: %[[TMP_12:.*]] = "mhlo.select"(%[[TMP_9]], %[[TMP_10]], %[[TMP_11]])
  // CHECK: %[[TMP_8:.*]] = mhlo.constant dense<0.99999999999980993>
  // CHECK: %[[TMP_13:.*]] = mhlo.constant dense<676.5203681218851>
  // CHECK: %[[TMP_14:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_12]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = mhlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = mhlo.add %[[TMP_8]], %[[TMP_16]]
  // CHECK: %[[TMP_18:.*]] = mhlo.constant dense<-1259.1392167224028>
  // CHECK: %[[TMP_19:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_12]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK: %[[TMP_23:.*]] = mhlo.constant dense<771.32342877765313>
  // CHECK: %[[TMP_24:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = mhlo.add %[[TMP_12]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = mhlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = mhlo.constant dense<-176.61502916214059>
  // CHECK: %[[TMP_29:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = mhlo.add %[[TMP_12]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = mhlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK: %[[TMP_33:.*]] = mhlo.constant dense<12.507343278686905>
  // CHECK: %[[TMP_34:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = mhlo.add %[[TMP_12]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = mhlo.constant dense<-0.13857109526572012>
  // CHECK: %[[TMP_39:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = mhlo.add %[[TMP_12]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = mhlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = mhlo.constant dense<9.9843695780195716E-6>
  // CHECK: %[[TMP_44:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = mhlo.add %[[TMP_12]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK: %[[TMP_48:.*]] = mhlo.constant dense<1.5056327351493116E-7>
  // CHECK: %[[TMP_49:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_12]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_53:.*]] = mhlo.add %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_7:.*]] = mhlo.constant dense<2.0149030205422647>
  // CHECK: %[[TMP_54:.*]] = mhlo.divide %[[TMP_12]], %[[TMP_6]]
  // CHECK: %[[TMP_55:.*]] = "mhlo.log_plus_one"(%[[TMP_54]])
  // CHECK: %[[TMP_56:.*]] = mhlo.add %[[TMP_7]], %[[TMP_55]]
  // CHECK: %[[TMP_57:.*]] = mhlo.divide %[[TMP_53]], %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_12]], %[[TMP_1]]
  // CHECK: %[[TMP_59:.*]] = mhlo.subtract %[[TMP_58]], %[[TMP_57]]
  // CHECK: %[[TMP_60:.*]] = mhlo.multiply %[[TMP_59]], %[[TMP_56]]
  // CHECK: %[[TMP_61:.*]] = "mhlo.log"(%[[TMP_52]])
  // CHECK: %[[TMP_5:.*]] = mhlo.constant dense<0.91893853320467266>
  // CHECK: %[[TMP_62:.*]] = mhlo.add %[[TMP_5]], %[[TMP_60]]
  // CHECK: %[[TMP_63:.*]] = mhlo.add %[[TMP_62]], %[[TMP_61]]
  // CHECK: %[[TMP_64:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_65:.*]] = "mhlo.floor"(%[[TMP_64]])
  // CHECK: %[[TMP_66:.*]] = mhlo.subtract %[[TMP_64]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = "mhlo.compare"(%[[TMP_1]], %[[TMP_66]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_68:.*]] = mhlo.subtract %[[TMP_2]], %[[TMP_66]]
  // CHECK: %[[TMP_69:.*]] = "mhlo.select"(%[[TMP_67]], %[[TMP_68]], %[[TMP_66]])
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_70:.*]] = mhlo.multiply %[[TMP_3]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = "mhlo.sine"(%[[TMP_70]])
  // CHECK: %[[TMP_72:.*]] = "mhlo.log"(%[[TMP_71]])
  // CHECK: %[[TMP_4:.*]] = mhlo.constant dense<1.1447298858494002>
  // CHECK: %[[TMP_75:.*]] = mhlo.subtract %[[TMP_4]], %[[TMP_72]]
  // CHECK: %[[TMP_76:.*]] = mhlo.subtract %[[TMP_75]], %[[TMP_63]]
  // CHECK: %[[TMP_73:.*]] = "mhlo.is_finite"(%[[TMP_72]])
  // CHECK: %[[TMP_74:.*]] = "mhlo.negate"(%[[TMP_72]])
  // CHECK: %[[TMP_77:.*]] = "mhlo.select"(%[[TMP_73]], %[[TMP_76]], %[[TMP_74]])
  // CHECK: %[[TMP_78:.*]] = "mhlo.select"(%[[TMP_9]], %[[TMP_77]], %[[TMP_63]])
  // CHECK: %[[TMP_79:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_80:.*]] = mhlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_81:.*]] = "mhlo.compare"(%[[TMP_79]], %[[TMP_80]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_0:.*]] = mhlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_82:.*]] = "mhlo.select"(%[[TMP_81]], %[[TMP_0]], %[[TMP_78]])
  // CHECK: return %[[TMP_82]]
  %1 = chlo.lgamma %arg : tensor<f64> -> tensor<f64>
  return %1 : tensor<f64>
}

// ----

// CHECK-LABEL: @lgamma_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func @lgamma_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_1:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_9:.*]] = "mhlo.compare"(%[[ARG]], %[[TMP_1]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_10:.*]] = "mhlo.negate"(%[[ARG]])
  // CHECK: %[[TMP_2:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_11:.*]] = mhlo.subtract %[[ARG]], %[[TMP_2]]
  // CHECK: %[[TMP_12:.*]] = "mhlo.select"(%[[TMP_9]], %[[TMP_10]], %[[TMP_11]])
  // CHECK: %[[TMP_8:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_13:.*]] = mhlo.constant dense<676.520386>
  // CHECK: %[[TMP_14:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_12]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = mhlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = mhlo.add %[[TMP_8]], %[[TMP_16]]
  // CHECK: %[[TMP_18:.*]] = mhlo.constant dense<-1259.13916>
  // CHECK: %[[TMP_19:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_12]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK: %[[TMP_23:.*]] = mhlo.constant dense<771.323425>
  // CHECK: %[[TMP_24:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = mhlo.add %[[TMP_12]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = mhlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = mhlo.constant dense<-176.615036>
  // CHECK: %[[TMP_29:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = mhlo.add %[[TMP_12]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = mhlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK: %[[TMP_33:.*]] = mhlo.constant dense<12.5073433>
  // CHECK: %[[TMP_34:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = mhlo.add %[[TMP_12]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = mhlo.constant dense<-0.138571098>
  // CHECK: %[[TMP_39:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = mhlo.add %[[TMP_12]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = mhlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = mhlo.constant dense<9.98436917E-6>
  // CHECK: %[[TMP_44:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = mhlo.add %[[TMP_12]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK: %[[TMP_48:.*]] = mhlo.constant dense<1.50563267E-7>
  // CHECK: %[[TMP_49:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_12]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_53:.*]] = mhlo.add %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_7:.*]] = mhlo.constant dense<2.01490307>
  // CHECK: %[[TMP_54:.*]] = mhlo.divide %[[TMP_12]], %[[TMP_6]]
  // CHECK: %[[TMP_55:.*]] = "mhlo.log_plus_one"(%[[TMP_54]])
  // CHECK: %[[TMP_56:.*]] = mhlo.add %[[TMP_7]], %[[TMP_55]]
  // CHECK: %[[TMP_57:.*]] = mhlo.divide %[[TMP_53]], %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_12]], %[[TMP_1]]
  // CHECK: %[[TMP_59:.*]] = mhlo.subtract %[[TMP_58]], %[[TMP_57]]
  // CHECK: %[[TMP_60:.*]] = mhlo.multiply %[[TMP_59]], %[[TMP_56]]
  // CHECK: %[[TMP_61:.*]] = "mhlo.log"(%[[TMP_52]])
  // CHECK: %[[TMP_5:.*]] = mhlo.constant dense<0.918938517>
  // CHECK: %[[TMP_62:.*]] = mhlo.add %[[TMP_5]], %[[TMP_60]]
  // CHECK: %[[TMP_63:.*]] = mhlo.add %[[TMP_62]], %[[TMP_61]]
  // CHECK: %[[TMP_64:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_65:.*]] = "mhlo.floor"(%[[TMP_64]])
  // CHECK: %[[TMP_66:.*]] = mhlo.subtract %[[TMP_64]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = "mhlo.compare"(%[[TMP_1]], %[[TMP_66]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_68:.*]] = mhlo.subtract %[[TMP_2]], %[[TMP_66]]
  // CHECK: %[[TMP_69:.*]] = "mhlo.select"(%[[TMP_67]], %[[TMP_68]], %[[TMP_66]])
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<3.14159274>
  // CHECK: %[[TMP_70:.*]] = mhlo.multiply %[[TMP_3]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = "mhlo.sine"(%[[TMP_70]])
  // CHECK: %[[TMP_72:.*]] = "mhlo.log"(%[[TMP_71]])
  // CHECK: %[[TMP_4:.*]] = mhlo.constant dense<1.14472985>
  // CHECK: %[[TMP_75:.*]] = mhlo.subtract %[[TMP_4]], %[[TMP_72]]
  // CHECK: %[[TMP_76:.*]] = mhlo.subtract %[[TMP_75]], %[[TMP_63]]
  // CHECK: %[[TMP_73:.*]] = "mhlo.is_finite"(%[[TMP_72]])
  // CHECK: %[[TMP_74:.*]] = "mhlo.negate"(%[[TMP_72]])
  // CHECK: %[[TMP_77:.*]] = "mhlo.select"(%[[TMP_73]], %[[TMP_76]], %[[TMP_74]])
  // CHECK: %[[TMP_78:.*]] = "mhlo.select"(%[[TMP_9]], %[[TMP_77]], %[[TMP_63]])
  // CHECK: %[[TMP_79:.*]] = "mhlo.abs"(%[[ARG]])
  // CHECK: %[[TMP_80:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_81:.*]] = "mhlo.compare"(%[[TMP_79]], %[[TMP_80]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_0:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_82:.*]] = "mhlo.select"(%[[TMP_81]], %[[TMP_0]], %[[TMP_78]])
  // CHECK: return %[[TMP_82]]
  %1 = chlo.lgamma %arg : tensor<f32> -> tensor<f32>
  return %1 : tensor<f32>
}

// ----

// CHECK-LABEL: @lgamma_f16
// CHECK-SAME: (%[[ARG:.*]]: tensor<f16>)
func @lgamma_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.convert"(%[[ARG]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = "mhlo.convert"(%{{.*}}) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.lgamma %arg : tensor<f16> -> tensor<f16>
  return %1 : tensor<f16>
}

// ----

// CHECK-LABEL: @digamma_f64
// CHECK-SAME: (%[[ARG:.*]]: tensor<f64>)
func @digamma_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK: %[[TMP_0:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_1:.*]] = "mhlo.compare"(%arg0, %[[TMP_0]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_2:.*]] = "mhlo.negate"(%arg0)
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_4:.*]] = mhlo.subtract %arg0, %[[TMP_3]]
  // CHECK: %[[TMP_5:.*]] = "mhlo.select"(%[[TMP_1]], %[[TMP_2]], %[[TMP_4]])
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_7:.*]] = mhlo.constant dense<0.99999999999980993>
  // CHECK: %[[TMP_8:.*]] = mhlo.constant dense<676.5203681218851>
  // CHECK: %[[TMP_9:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = mhlo.add %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = mhlo.multiply %[[TMP_10]], %[[TMP_10]]
  // CHECK: %[[TMP_12:.*]] = mhlo.divide %[[TMP_8]], %[[TMP_11]]
  // CHECK: %[[TMP_13:.*]] = mhlo.subtract %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_14:.*]] = mhlo.divide %[[TMP_8]], %[[TMP_10]]
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_7]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = mhlo.constant dense<-1259.1392167224028>
  // CHECK: %[[TMP_17:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_18:.*]] = mhlo.add %[[TMP_5]], %[[TMP_17]]
  // CHECK: %[[TMP_19:.*]] = mhlo.multiply %[[TMP_18]], %[[TMP_18]]
  // CHECK: %[[TMP_20:.*]] = mhlo.divide %[[TMP_16]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.subtract %[[TMP_13]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.divide %[[TMP_16]], %[[TMP_18]]
  // CHECK: %[[TMP_23:.*]] = mhlo.add %[[TMP_15]], %[[TMP_22]]
  // CHECK: %[[TMP_24:.*]] = mhlo.constant dense<771.32342877765313>
  // CHECK: %[[TMP_25:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_26:.*]] = mhlo.add %[[TMP_5]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.multiply %[[TMP_26]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = mhlo.divide %[[TMP_24]], %[[TMP_27]]
  // CHECK: %[[TMP_29:.*]] = mhlo.subtract %[[TMP_21]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = mhlo.divide %[[TMP_24]], %[[TMP_26]]
  // CHECK: %[[TMP_31:.*]] = mhlo.add %[[TMP_23]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = mhlo.constant dense<-176.61502916214059>
  // CHECK: %[[TMP_33:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_34:.*]] = mhlo.add %[[TMP_5]], %[[TMP_33]]
  // CHECK: %[[TMP_35:.*]] = mhlo.multiply %[[TMP_34]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_32]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.subtract %[[TMP_29]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = mhlo.divide %[[TMP_32]], %[[TMP_34]]
  // CHECK: %[[TMP_39:.*]] = mhlo.add %[[TMP_31]], %[[TMP_38]]
  // CHECK: %[[TMP_40:.*]] = mhlo.constant dense<12.507343278686905>
  // CHECK: %[[TMP_41:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_5]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = mhlo.multiply %[[TMP_42]], %[[TMP_42]]
  // CHECK: %[[TMP_44:.*]] = mhlo.divide %[[TMP_40]], %[[TMP_43]]
  // CHECK: %[[TMP_45:.*]] = mhlo.subtract %[[TMP_37]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_40]], %[[TMP_42]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_39]], %[[TMP_46]]
  // CHECK: %[[TMP_48:.*]] = mhlo.constant dense<-0.13857109526572012>
  // CHECK: %[[TMP_49:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_5]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.multiply %[[TMP_50]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = mhlo.subtract %[[TMP_45]], %[[TMP_52]]
  // CHECK: %[[TMP_54:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_55:.*]] = mhlo.add %[[TMP_47]], %[[TMP_54]]
  // CHECK: %[[TMP_56:.*]] = mhlo.constant dense<9.9843695780195716E-6>
  // CHECK: %[[TMP_57:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_5]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = mhlo.multiply %[[TMP_58]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = mhlo.divide %[[TMP_56]], %[[TMP_59]]
  // CHECK: %[[TMP_61:.*]] = mhlo.subtract %[[TMP_53]], %[[TMP_60]]
  // CHECK: %[[TMP_62:.*]] = mhlo.divide %[[TMP_56]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = mhlo.add %[[TMP_55]], %[[TMP_62]]
  // CHECK: %[[TMP_64:.*]] = mhlo.constant dense<1.5056327351493116E-7>
  // CHECK: %[[TMP_65:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_66:.*]] = mhlo.add %[[TMP_5]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = mhlo.multiply %[[TMP_66]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = mhlo.divide %[[TMP_64]], %[[TMP_67]]
  // CHECK: %[[TMP_69:.*]] = mhlo.subtract %[[TMP_61]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = mhlo.divide %[[TMP_64]], %[[TMP_66]]
  // CHECK: %[[TMP_71:.*]] = mhlo.add %[[TMP_63]], %[[TMP_70]]
  // CHECK: %[[TMP_72:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_73:.*]] = mhlo.add %[[TMP_72]], %[[TMP_5]]
  // CHECK: %[[TMP_74:.*]] = mhlo.constant dense<2.0149030205422647>
  // CHECK: %[[TMP_75:.*]] = mhlo.divide %[[TMP_5]], %[[TMP_72]]
  // CHECK: %[[TMP_76:.*]] = "mhlo.log_plus_one"(%[[TMP_75]])
  // CHECK: %[[TMP_77:.*]] = mhlo.add %[[TMP_74]], %[[TMP_76]]
  // CHECK: %[[TMP_78:.*]] = mhlo.divide %[[TMP_69]], %[[TMP_71]]
  // CHECK: %[[TMP_79:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_80:.*]] = mhlo.divide %[[TMP_79]], %[[TMP_73]]
  // CHECK: %[[TMP_81:.*]] = mhlo.add %[[TMP_77]], %[[TMP_78]]
  // CHECK: %[[TMP_82:.*]] = mhlo.subtract %[[TMP_81]], %[[TMP_80]]
  // CHECK: %[[TMP_83:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_84:.*]] = mhlo.add %arg0, %[[TMP_83]]
  // CHECK: %[[TMP_85:.*]] = "mhlo.floor"(%[[TMP_84]])
  // CHECK: %[[TMP_86:.*]] = "mhlo.abs"(%[[TMP_85]])
  // CHECK: %[[TMP_87:.*]] = mhlo.add %arg0, %[[TMP_86]]
  // CHECK: %[[TMP_88:.*]] = mhlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_89:.*]] = mhlo.multiply %[[TMP_88]], %[[TMP_87]]
  // CHECK: %[[TMP_90:.*]] = "mhlo.cosine"(%[[TMP_89]])
  // CHECK: %[[TMP_92:.*]] = "mhlo.sine"(%[[TMP_89]])
  // CHECK: %[[TMP_91:.*]] = mhlo.multiply %[[TMP_88]], %[[TMP_90]]
  // CHECK: %[[TMP_93:.*]] = mhlo.divide %[[TMP_91]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = mhlo.subtract %[[TMP_82]], %[[TMP_93]]
  // CHECK: %[[TMP_95:.*]] = "mhlo.select"(%[[TMP_1]], %[[TMP_94]], %[[TMP_82]])
  // CHECK: %[[TMP_96:.*]] = "mhlo.compare"(%arg0, %[[TMP_6]]) {comparison_direction = "LE"}
  // CHECK: %[[TMP_97:.*]] = "mhlo.floor"(%arg0)
  // CHECK: %[[TMP_98:.*]] = "mhlo.compare"(%arg0, %[[TMP_97]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_99:.*]] = mhlo.and %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[RES:.*]] = "mhlo.select"(%[[TMP_99]], %[[TMP_100]], %[[TMP_95]])
  // CHECK: return %[[RES]]
  %1 = chlo.digamma %arg : tensor<f64> -> tensor<f64>
  return %1 : tensor<f64>
}

// ----

// CHECK-LABEL: @digamma_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func @digamma_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_0:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_1:.*]] = "mhlo.compare"(%arg0, %[[TMP_0]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_2:.*]] = "mhlo.negate"(%arg0)
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_4:.*]] = mhlo.subtract %arg0, %[[TMP_3]]
  // CHECK: %[[TMP_5:.*]] = "mhlo.select"(%[[TMP_1]], %[[TMP_2]], %[[TMP_4]])
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_7:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_8:.*]] = mhlo.constant dense<676.520386>
  // CHECK: %[[TMP_9:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = mhlo.add %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = mhlo.multiply %[[TMP_10]], %[[TMP_10]]
  // CHECK: %[[TMP_12:.*]] = mhlo.divide %[[TMP_8]], %[[TMP_11]]
  // CHECK: %[[TMP_13:.*]] = mhlo.subtract %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_14:.*]] = mhlo.divide %[[TMP_8]], %[[TMP_10]]
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_7]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = mhlo.constant dense<-1259.13916>
  // CHECK: %[[TMP_17:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_18:.*]] = mhlo.add %[[TMP_5]], %[[TMP_17]]
  // CHECK: %[[TMP_19:.*]] = mhlo.multiply %[[TMP_18]], %[[TMP_18]]
  // CHECK: %[[TMP_20:.*]] = mhlo.divide %[[TMP_16]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.subtract %[[TMP_13]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.divide %[[TMP_16]], %[[TMP_18]]
  // CHECK: %[[TMP_23:.*]] = mhlo.add %[[TMP_15]], %[[TMP_22]]
  // CHECK: %[[TMP_24:.*]] = mhlo.constant dense<771.323425>
  // CHECK: %[[TMP_25:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_26:.*]] = mhlo.add %[[TMP_5]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.multiply %[[TMP_26]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = mhlo.divide %[[TMP_24]], %[[TMP_27]]
  // CHECK: %[[TMP_29:.*]] = mhlo.subtract %[[TMP_21]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = mhlo.divide %[[TMP_24]], %[[TMP_26]]
  // CHECK: %[[TMP_31:.*]] = mhlo.add %[[TMP_23]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = mhlo.constant dense<-176.615036>
  // CHECK: %[[TMP_33:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_34:.*]] = mhlo.add %[[TMP_5]], %[[TMP_33]]
  // CHECK: %[[TMP_35:.*]] = mhlo.multiply %[[TMP_34]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_32]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.subtract %[[TMP_29]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = mhlo.divide %[[TMP_32]], %[[TMP_34]]
  // CHECK: %[[TMP_39:.*]] = mhlo.add %[[TMP_31]], %[[TMP_38]]
  // CHECK: %[[TMP_40:.*]] = mhlo.constant dense<12.5073433>
  // CHECK: %[[TMP_41:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_5]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = mhlo.multiply %[[TMP_42]], %[[TMP_42]]
  // CHECK: %[[TMP_44:.*]] = mhlo.divide %[[TMP_40]], %[[TMP_43]]
  // CHECK: %[[TMP_45:.*]] = mhlo.subtract %[[TMP_37]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_40]], %[[TMP_42]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_39]], %[[TMP_46]]
  // CHECK: %[[TMP_48:.*]] = mhlo.constant dense<-0.138571098>
  // CHECK: %[[TMP_49:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_5]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.multiply %[[TMP_50]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = mhlo.subtract %[[TMP_45]], %[[TMP_52]]
  // CHECK: %[[TMP_54:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_55:.*]] = mhlo.add %[[TMP_47]], %[[TMP_54]]
  // CHECK: %[[TMP_56:.*]] = mhlo.constant dense<9.98436917E-6>
  // CHECK: %[[TMP_57:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_5]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = mhlo.multiply %[[TMP_58]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = mhlo.divide %[[TMP_56]], %[[TMP_59]]
  // CHECK: %[[TMP_61:.*]] = mhlo.subtract %[[TMP_53]], %[[TMP_60]]
  // CHECK: %[[TMP_62:.*]] = mhlo.divide %[[TMP_56]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = mhlo.add %[[TMP_55]], %[[TMP_62]]
  // CHECK: %[[TMP_64:.*]] = mhlo.constant dense<1.50563267E-7>
  // CHECK: %[[TMP_65:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_66:.*]] = mhlo.add %[[TMP_5]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = mhlo.multiply %[[TMP_66]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = mhlo.divide %[[TMP_64]], %[[TMP_67]]
  // CHECK: %[[TMP_69:.*]] = mhlo.subtract %[[TMP_61]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = mhlo.divide %[[TMP_64]], %[[TMP_66]]
  // CHECK: %[[TMP_71:.*]] = mhlo.add %[[TMP_63]], %[[TMP_70]]
  // CHECK: %[[TMP_72:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_73:.*]] = mhlo.add %[[TMP_72]], %[[TMP_5]]
  // CHECK: %[[TMP_74:.*]] = mhlo.constant dense<2.01490307>
  // CHECK: %[[TMP_75:.*]] = mhlo.divide %[[TMP_5]], %[[TMP_72]]
  // CHECK: %[[TMP_76:.*]] = "mhlo.log_plus_one"(%[[TMP_75]])
  // CHECK: %[[TMP_77:.*]] = mhlo.add %[[TMP_74]], %[[TMP_76]]
  // CHECK: %[[TMP_78:.*]] = mhlo.divide %[[TMP_69]], %[[TMP_71]]
  // CHECK: %[[TMP_79:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_80:.*]] = mhlo.divide %[[TMP_79]], %[[TMP_73]]
  // CHECK: %[[TMP_81:.*]] = mhlo.add %[[TMP_77]], %[[TMP_78]]
  // CHECK: %[[TMP_82:.*]] = mhlo.subtract %[[TMP_81]], %[[TMP_80]]
  // CHECK: %[[TMP_83:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_84:.*]] = mhlo.add %arg0, %[[TMP_83]]
  // CHECK: %[[TMP_85:.*]] = "mhlo.floor"(%[[TMP_84]])
  // CHECK: %[[TMP_86:.*]] = "mhlo.abs"(%[[TMP_85]])
  // CHECK: %[[TMP_87:.*]] = mhlo.add %arg0, %[[TMP_86]]
  // CHECK: %[[TMP_88:.*]] = mhlo.constant dense<3.14159274>
  // CHECK: %[[TMP_89:.*]] = mhlo.multiply %[[TMP_88]], %[[TMP_87]]
  // CHECK: %[[TMP_90:.*]] = "mhlo.cosine"(%[[TMP_89]])
  // CHECK: %[[TMP_92:.*]] = "mhlo.sine"(%[[TMP_89]])
  // CHECK: %[[TMP_91:.*]] = mhlo.multiply %[[TMP_88]], %[[TMP_90]]
  // CHECK: %[[TMP_93:.*]] = mhlo.divide %[[TMP_91]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = mhlo.subtract %[[TMP_82]], %[[TMP_93]]
  // CHECK: %[[TMP_95:.*]] = "mhlo.select"(%[[TMP_1]], %[[TMP_94]], %[[TMP_82]])
  // CHECK: %[[TMP_96:.*]] = "mhlo.compare"(%arg0, %[[TMP_6]]) {comparison_direction = "LE"}
  // CHECK: %[[TMP_97:.*]] = "mhlo.floor"(%arg0)
  // CHECK: %[[TMP_98:.*]] = "mhlo.compare"(%arg0, %[[TMP_97]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_99:.*]] = mhlo.and %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.constant dense<0x7FC00000>
  // CHECK: %[[RES:.*]] = "mhlo.select"(%[[TMP_99]], %[[TMP_100]], %[[TMP_95]])
  // CHECK: return %[[RES]]
  %1 = chlo.digamma %arg : tensor<f32> -> tensor<f32>
  return %1 : tensor<f32>
}

// ----

// CHECK-LABEL: @digamma_f16
// CHECK-SAME: (%[[ARG:.*]]: tensor<f16>)
func @digamma_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.convert"(%[[ARG]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = "mhlo.convert"(%{{.*}}) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.digamma %arg : tensor<f16> -> tensor<f16>
  return %1 : tensor<f16>
}

// ----

// CHECK-LABEL: @zeta_f16
// CHECK-SAME:  (%[[X:.*]]: tensor<f16>, %[[Q:.*]]: tensor<f16>) -> tensor<f16>
func @zeta_f16(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
  // CHECK: %[[TMP_0:.*]] = "mhlo.convert"(%[[X]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[TMP_1:.*]] = "mhlo.convert"(%[[Q]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[TMP_2:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_3:.*]] = "mhlo.negate"(%[[TMP_0]])
  // CHECK: %[[TMP_4:.*]] = mhlo.power %[[TMP_1]], %[[TMP_3]]
  // CHECK: %[[TMP_5:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_6:.*]] = mhlo.add %[[TMP_1]], %[[TMP_5]]
  // CHECK: %[[TMP_7:.*]] = mhlo.power %[[TMP_6]], %[[TMP_3]]
  // CHECK: %[[TMP_8:.*]] = mhlo.add %[[TMP_4]], %[[TMP_7]]
  // CHECK: %[[TMP_9:.*]] = mhlo.add %[[TMP_6]], %[[TMP_5]]
  // CHECK: %[[TMP_10:.*]] = mhlo.power %[[TMP_9]], %[[TMP_3]]
  // CHECK: %[[TMP_11:.*]] = mhlo.add %[[TMP_8]], %[[TMP_10]]
  // CHECK: %[[TMP_12:.*]] = mhlo.add %[[TMP_9]], %[[TMP_5]]
  // CHECK: %[[TMP_13:.*]] = mhlo.power %[[TMP_12]], %[[TMP_3]]
  // CHECK: %[[TMP_14:.*]] = mhlo.add %[[TMP_11]], %[[TMP_13]]
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_12]], %[[TMP_5]]
  // CHECK: %[[TMP_16:.*]] = mhlo.power %[[TMP_15]], %[[TMP_3]]
  // CHECK: %[[TMP_17:.*]] = mhlo.add %[[TMP_14]], %[[TMP_16]]
  // CHECK: %[[TMP_18:.*]] = mhlo.add %[[TMP_15]], %[[TMP_5]]
  // CHECK: %[[TMP_19:.*]] = mhlo.power %[[TMP_18]], %[[TMP_3]]
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_17]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.add %[[TMP_18]], %[[TMP_5]]
  // CHECK: %[[TMP_22:.*]] = mhlo.power %[[TMP_21]], %[[TMP_3]]
  // CHECK: %[[TMP_23:.*]] = mhlo.add %[[TMP_20]], %[[TMP_22]]
  // CHECK: %[[TMP_24:.*]] = mhlo.add %[[TMP_21]], %[[TMP_5]]
  // CHECK: %[[TMP_25:.*]] = mhlo.power %[[TMP_24]], %[[TMP_3]]
  // CHECK: %[[TMP_26:.*]] = mhlo.add %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.add %[[TMP_24]], %[[TMP_5]]
  // CHECK: %[[TMP_28:.*]] = mhlo.power %[[TMP_27]], %[[TMP_3]]
  // CHECK: %[[TMP_29:.*]] = mhlo.add %[[TMP_26]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = mhlo.add %[[TMP_27]], %[[TMP_5]]
  // CHECK: %[[TMP_31:.*]] = mhlo.power %[[TMP_30]], %[[TMP_3]]
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_29]], %[[TMP_31]]
  // CHECK: %[[TMP_33:.*]] = mhlo.add %[[TMP_30]], %[[TMP_5]]
  // CHECK: %[[TMP_34:.*]] = mhlo.power %[[TMP_33]], %[[TMP_3]]
  // CHECK: %[[TMP_35:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_36:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.multiply %[[TMP_34]], %[[TMP_33]]
  // CHECK: %[[TMP_38:.*]] = mhlo.divide %[[TMP_37]], %[[TMP_36]]
  // CHECK: %[[TMP_39:.*]] = mhlo.add %[[TMP_32]], %[[TMP_38]]
  // CHECK: %[[TMP_40:.*]] = mhlo.multiply %[[TMP_33]], %[[TMP_33]]
  // CHECK: %[[TMP_41:.*]] = mhlo.divide %[[TMP_5]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = mhlo.constant dense<2.200000e+01>
  // CHECK: %[[TMP_43:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_42]]
  // CHECK: %[[TMP_44:.*]] = mhlo.constant dense<2.100000e+01>
  // CHECK: %[[TMP_45:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.multiply %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = mhlo.constant dense<-1.39544646E-19>
  // CHECK: %[[TMP_48:.*]] = mhlo.add %[[TMP_2]], %[[TMP_47]]
  // CHECK: %[[TMP_49:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_48]]
  // CHECK: %[[TMP_50:.*]] = mhlo.multiply %[[TMP_46]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.constant dense<2.000000e+01>
  // CHECK: %[[TMP_52:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = mhlo.constant dense<1.900000e+01>
  // CHECK: %[[TMP_54:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_53]]
  // CHECK: %[[TMP_55:.*]] = mhlo.multiply %[[TMP_52]], %[[TMP_54]]
  // CHECK: %[[TMP_56:.*]] = mhlo.constant dense<5.50900303E-18>
  // CHECK: %[[TMP_57:.*]] = mhlo.add %[[TMP_50]], %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = mhlo.multiply %[[TMP_55]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = mhlo.constant dense<1.800000e+01>
  // CHECK: %[[TMP_61:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_60]]
  // CHECK: %[[TMP_62:.*]] = mhlo.constant dense<1.700000e+01>
  // CHECK: %[[TMP_63:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_62]]
  // CHECK: %[[TMP_64:.*]] = mhlo.multiply %[[TMP_61]], %[[TMP_63]]
  // CHECK: %[[TMP_65:.*]] = mhlo.constant dense<-2.17486866E-16>
  // CHECK: %[[TMP_66:.*]] = mhlo.add %[[TMP_59]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = mhlo.multiply %[[TMP_64]], %[[TMP_67]]
  // CHECK: %[[TMP_69:.*]] = mhlo.constant dense<1.600000e+01>
  // CHECK: %[[TMP_70:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = mhlo.constant dense<1.500000e+01>
  // CHECK: %[[TMP_72:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_71]]
  // CHECK: %[[TMP_73:.*]] = mhlo.multiply %[[TMP_70]], %[[TMP_72]]
  // CHECK: %[[TMP_74:.*]] = mhlo.constant dense<8.58606213E-15>
  // CHECK: %[[TMP_75:.*]] = mhlo.add %[[TMP_68]], %[[TMP_74]]
  // CHECK: %[[TMP_76:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_75]]
  // CHECK: %[[TMP_77:.*]] = mhlo.multiply %[[TMP_73]], %[[TMP_76]]
  // CHECK: %[[TMP_78:.*]] = mhlo.constant dense<1.400000e+01>
  // CHECK: %[[TMP_79:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_78]]
  // CHECK: %[[TMP_80:.*]] = mhlo.constant dense<1.300000e+01>
  // CHECK: %[[TMP_81:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_80]]
  // CHECK: %[[TMP_82:.*]] = mhlo.multiply %[[TMP_79]], %[[TMP_81]]
  // CHECK: %[[TMP_83:.*]] = mhlo.constant dense<-3.3896803E-13>
  // CHECK: %[[TMP_84:.*]] = mhlo.add %[[TMP_77]], %[[TMP_83]]
  // CHECK: %[[TMP_85:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_84]]
  // CHECK: %[[TMP_86:.*]] = mhlo.multiply %[[TMP_82]], %[[TMP_85]]
  // CHECK: %[[TMP_87:.*]] = mhlo.constant dense<1.200000e+01>
  // CHECK: %[[TMP_88:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_87]]
  // CHECK: %[[TMP_89:.*]] = mhlo.constant dense<1.100000e+01>
  // CHECK: %[[TMP_90:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_89]]
  // CHECK: %[[TMP_91:.*]] = mhlo.multiply %[[TMP_88]], %[[TMP_90]]
  // CHECK: %[[TMP_92:.*]] = mhlo.constant dense<1.33825364E-11>
  // CHECK: %[[TMP_93:.*]] = mhlo.add %[[TMP_86]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_93]]
  // CHECK: %[[TMP_95:.*]] = mhlo.multiply %[[TMP_91]], %[[TMP_94]]
  // CHECK: %[[TMP_96:.*]] = mhlo.constant dense<1.000000e+01>
  // CHECK: %[[TMP_97:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_96]]
  // CHECK: %[[TMP_98:.*]] = mhlo.constant dense<9.000000e+00>
  // CHECK: %[[TMP_99:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.multiply %[[TMP_97]], %[[TMP_99]]
  // CHECK: %[[TMP_101:.*]] = mhlo.constant dense<-5.28419031E-10>
  // CHECK: %[[TMP_102:.*]] = mhlo.add %[[TMP_95]], %[[TMP_101]]
  // CHECK: %[[TMP_103:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_102]]
  // CHECK: %[[TMP_104:.*]] = mhlo.multiply %[[TMP_100]], %[[TMP_103]]
  // CHECK: %[[TMP_105:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_106:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_105]]
  // CHECK: %[[TMP_107:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_108:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_107]]
  // CHECK: %[[TMP_109:.*]] = mhlo.multiply %[[TMP_106]], %[[TMP_108]]
  // CHECK: %[[TMP_110:.*]] = mhlo.constant dense<2.08767563E-8>
  // CHECK: %[[TMP_111:.*]] = mhlo.add %[[TMP_104]], %[[TMP_110]]
  // CHECK: %[[TMP_112:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_111]]
  // CHECK: %[[TMP_113:.*]] = mhlo.multiply %[[TMP_109]], %[[TMP_112]]
  // CHECK: %[[TMP_114:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_115:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_114]]
  // CHECK: %[[TMP_116:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_117:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_116]]
  // CHECK: %[[TMP_118:.*]] = mhlo.multiply %[[TMP_115]], %[[TMP_117]]
  // CHECK: %[[TMP_119:.*]] = mhlo.constant dense<-8.26719599E-7>
  // CHECK: %[[TMP_120:.*]] = mhlo.add %[[TMP_113]], %[[TMP_119]]
  // CHECK: %[[TMP_121:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_120]]
  // CHECK: %[[TMP_122:.*]] = mhlo.multiply %[[TMP_118]], %[[TMP_121]]
  // CHECK: %[[TMP_123:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_124:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_123]]
  // CHECK: %[[TMP_125:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_126:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_125]]
  // CHECK: %[[TMP_127:.*]] = mhlo.multiply %[[TMP_124]], %[[TMP_126]]
  // CHECK: %[[TMP_128:.*]] = mhlo.constant dense<3.30687835E-5>
  // CHECK: %[[TMP_129:.*]] = mhlo.add %[[TMP_122]], %[[TMP_128]]
  // CHECK: %[[TMP_130:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_129]]
  // CHECK: %[[TMP_131:.*]] = mhlo.multiply %[[TMP_127]], %[[TMP_130]]
  // CHECK: %[[TMP_132:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_133:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_132]]
  // CHECK: %[[TMP_134:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_135:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_134]]
  // CHECK: %[[TMP_136:.*]] = mhlo.multiply %[[TMP_133]], %[[TMP_135]]
  // CHECK: %[[TMP_137:.*]] = mhlo.constant dense<-0.00138888892>
  // CHECK: %[[TMP_138:.*]] = mhlo.add %[[TMP_131]], %[[TMP_137]]
  // CHECK: %[[TMP_139:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_138]]
  // CHECK: %[[TMP_140:.*]] = mhlo.multiply %[[TMP_136]], %[[TMP_139]]
  // CHECK: %[[TMP_141:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_142:.*]] = mhlo.divide %[[TMP_0]], %[[TMP_33]]
  // CHECK: %[[TMP_143:.*]] = mhlo.constant dense<0.0833333358>
  // CHECK: %[[TMP_144:.*]] = mhlo.add %[[TMP_143]], %[[TMP_140]]
  // CHECK: %[[TMP_145:.*]] = mhlo.multiply %[[TMP_142]], %[[TMP_144]]
  // CHECK: %[[TMP_146:.*]] = mhlo.add %[[TMP_141]], %[[TMP_145]]
  // CHECK: %[[TMP_147:.*]] = mhlo.multiply %[[TMP_34]], %[[TMP_146]]
  // CHECK: %[[TMP_148:.*]] = mhlo.add %[[TMP_39]], %[[TMP_147]]
  // CHECK: %[[TMP_149:.*]] = "mhlo.abs"(%[[TMP_34]])
  // CHECK: %[[TMP_150:.*]] = "mhlo.abs"(%[[TMP_32]])
  // CHECK: %[[TMP_151:.*]] = mhlo.constant dense<1.401300e-45>
  // CHECK: %[[TMP_152:.*]] = mhlo.multiply %[[TMP_150]], %[[TMP_151]]
  // CHECK: %[[TMP_153:.*]] = "mhlo.compare"(%[[TMP_149]], %[[TMP_152]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_154:.*]] = "mhlo.select"(%[[TMP_153]], %[[TMP_32]], %[[TMP_148]])
  // CHECK: %[[TMP_155:.*]] = mhlo.constant dense<0x7FC00000>
  // CHECK: %[[TMP_156:.*]] = "mhlo.compare"(%[[TMP_0]], %[[TMP_35]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_157:.*]] = "mhlo.select"(%[[TMP_156]], %[[TMP_155]], %[[TMP_154]])
  // CHECK: %[[TMP_158:.*]] = "mhlo.compare"(%[[TMP_1]], %[[TMP_2]]) {comparison_direction = "LE"}
  // CHECK: %[[TMP_159:.*]] = "mhlo.floor"(%[[TMP_0]])
  // CHECK: %[[TMP_160:.*]] = "mhlo.compare"(%[[TMP_0]], %[[TMP_159]]) {comparison_direction = "NE"}
  // CHECK: %[[TMP_161:.*]] = mhlo.and %[[TMP_158]], %[[TMP_160]] : tensor<i1>
  // CHECK: %[[TMP_162:.*]] = "mhlo.select"(%[[TMP_161]], %[[TMP_155]], %[[TMP_157]])
  // CHECK: %[[TMP_163:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_164:.*]] = "mhlo.floor"(%[[TMP_1]])
  // CHECK: %[[TMP_165:.*]] = "mhlo.compare"(%[[TMP_1]], %[[TMP_164]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_166:.*]] = mhlo.and %[[TMP_158]], %[[TMP_165]] : tensor<i1>
  // CHECK: %[[TMP_167:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_168:.*]] = "mhlo.floor"(%[[TMP_0]])
  // CHECK: %[[TMP_169:.*]] = "mhlo.compare"(%[[TMP_0]], %[[TMP_168]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_170:.*]] = mhlo.remainder %[[TMP_0]], %[[TMP_167]]
  // CHECK: %[[TMP_171:.*]] = "mhlo.compare"(%[[TMP_170]], %[[TMP_2]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_172:.*]] = mhlo.and %[[TMP_169]], %[[TMP_171]] : tensor<i1>
  // CHECK: %[[TMP_173:.*]] = "mhlo.select"(%[[TMP_172]], %[[TMP_163]], %[[TMP_155]])
  // CHECK: %[[TMP_174:.*]] = "mhlo.select"(%[[TMP_166]], %[[TMP_173]], %[[TMP_162]])
  // CHECK: %[[TMP_175:.*]] = "mhlo.compare"(%[[TMP_0]], %[[TMP_5]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_176:.*]] = "mhlo.select"(%[[TMP_175]], %[[TMP_163]], %[[TMP_174]])
  // CHECK: %[[TMP_177:.*]] = "mhlo.convert"(%[[TMP_176]]) : (tensor<f32>) -> tensor<f16>
  %0 = chlo.zeta %arg0, %arg1 : tensor<f16>, tensor<f16> -> tensor<f16>
  return %0 : tensor<f16>
}

// ----

// CHECK: @polygamma_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>)
func @polygamma_f32(%lhs : tensor<f32>, %rhs : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_0:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_1:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_2:.*]] = mhlo.remainder %[[ARG0]], %[[TMP_1]]
  // CHECK: %[[TMP_3:.*]] = mhlo.multiply %[[TMP_1]], %[[TMP_2]]
  // CHECK: %[[TMP_4:.*]] = mhlo.subtract %[[TMP_3]], %[[TMP_0]]
  // CHECK: %[[TMP_5:.*]] = mhlo.add %[[ARG0]], %[[TMP_0]]
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_7:.*]] = "mhlo.compare"(%[[TMP_5]], %[[TMP_6]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_8:.*]] = "mhlo.negate"(%[[TMP_5]])
  // CHECK: %[[TMP_9:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = "mhlo.select"(%[[TMP_7]], %[[TMP_8]], %[[TMP_10]])
  // CHECK: %[[TMP_12:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_13:.*]] = mhlo.constant dense<676.520386>
  // CHECK: %[[TMP_14:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_11]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = mhlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = mhlo.add %[[TMP_12]], %[[TMP_16]]
  // CHECK: %[[TMP_18:.*]] = mhlo.constant dense<-1259.13916>
  // CHECK: %[[TMP_19:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_11]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK: %[[TMP_23:.*]] = mhlo.constant dense<771.323425>
  // CHECK: %[[TMP_24:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = mhlo.add %[[TMP_11]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = mhlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = mhlo.constant dense<-176.615036>
  // CHECK: %[[TMP_29:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = mhlo.add %[[TMP_11]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = mhlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK: %[[TMP_33:.*]] = mhlo.constant dense<12.5073433>
  // CHECK: %[[TMP_34:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = mhlo.add %[[TMP_11]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = mhlo.constant dense<-0.138571098>
  // CHECK: %[[TMP_39:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = mhlo.add %[[TMP_11]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = mhlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = mhlo.constant dense<9.98436917E-6>
  // CHECK: %[[TMP_44:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = mhlo.add %[[TMP_11]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK: %[[TMP_48:.*]] = mhlo.constant dense<1.50563267E-7>
  // CHECK: %[[TMP_49:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_11]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_54:.*]] = mhlo.add %[[TMP_53]], %[[TMP_11]]
  // CHECK: %[[TMP_55:.*]] = mhlo.constant dense<2.01490307>
  // CHECK: %[[TMP_56:.*]] = mhlo.divide %[[TMP_11]], %[[TMP_53]]
  // CHECK: %[[TMP_57:.*]] = "mhlo.log_plus_one"(%[[TMP_56]])
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_55]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = mhlo.divide %[[TMP_54]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = mhlo.add %[[TMP_11]], %[[TMP_6]]
  // CHECK: %[[TMP_61:.*]] = mhlo.subtract %[[TMP_60]], %[[TMP_59]]
  // CHECK: %[[TMP_62:.*]] = mhlo.multiply %[[TMP_61]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = "mhlo.log"(%[[TMP_52]])
  // CHECK: %[[TMP_64:.*]] = mhlo.constant dense<0.918938517>
  // CHECK: %[[TMP_65:.*]] = mhlo.add %[[TMP_64]], %[[TMP_62]]
  // CHECK: %[[TMP_66:.*]] = mhlo.add %[[TMP_65]], %[[TMP_63]]
  // CHECK: %[[TMP_67:.*]] = "mhlo.abs"(%[[TMP_5]])
  // CHECK: %[[TMP_68:.*]] = "mhlo.floor"(%[[TMP_67]])
  // CHECK: %[[TMP_69:.*]] = mhlo.subtract %[[TMP_67]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = "mhlo.compare"(%[[TMP_6]], %[[TMP_69]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_71:.*]] = mhlo.subtract %[[TMP_9]], %[[TMP_69]]
  // CHECK: %[[TMP_72:.*]] = "mhlo.select"(%[[TMP_70]], %[[TMP_71]], %[[TMP_69]])
  // CHECK: %[[TMP_73:.*]] = mhlo.constant dense<3.14159274>
  // CHECK: %[[TMP_74:.*]] = mhlo.multiply %[[TMP_73]], %[[TMP_72]]
  // CHECK: %[[TMP_75:.*]] = "mhlo.sine"(%[[TMP_74]])
  // CHECK: %[[TMP_76:.*]] = "mhlo.log"(%[[TMP_75]])
  // CHECK: %[[TMP_77:.*]] = mhlo.constant dense<1.14472985>
  // CHECK: %[[TMP_78:.*]] = mhlo.subtract %[[TMP_77]], %[[TMP_76]]
  // CHECK: %[[TMP_79:.*]] = mhlo.subtract %[[TMP_78]], %[[TMP_66]]
  // CHECK: %[[TMP_80:.*]] = "mhlo.is_finite"(%[[TMP_76]])
  // CHECK: %[[TMP_81:.*]] = "mhlo.negate"(%[[TMP_76]])
  // CHECK: %[[TMP_82:.*]] = "mhlo.select"(%[[TMP_80]], %[[TMP_79]], %[[TMP_81]])
  // CHECK: %[[TMP_83:.*]] = "mhlo.select"(%[[TMP_7]], %[[TMP_82]], %[[TMP_66]])
  // CHECK: %[[TMP_84:.*]] = "mhlo.abs"(%[[TMP_5]])
  // CHECK: %[[TMP_85:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_86:.*]] = "mhlo.compare"(%[[TMP_84]], %[[TMP_85]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_87:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_88:.*]] = "mhlo.select"(%[[TMP_86]], %[[TMP_87]], %[[TMP_83]])
  // CHECK: %[[TMP_89:.*]] = "mhlo.exponential"(%[[TMP_88]])
  // CHECK: %[[TMP_90:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_91:.*]] = "mhlo.negate"(%[[TMP_5]])
  // CHECK: %[[TMP_92:.*]] = mhlo.power %[[ARG1]], %[[TMP_91]]
  // CHECK: %[[TMP_93:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_94:.*]] = mhlo.add %[[ARG1]], %[[TMP_93]]
  // CHECK: %[[TMP_95:.*]] = mhlo.power %[[TMP_94]], %[[TMP_91]]
  // CHECK: %[[TMP_96:.*]] = mhlo.add %[[TMP_92]], %[[TMP_95]]
  // CHECK: %[[TMP_97:.*]] = mhlo.add %[[TMP_94]], %[[TMP_93]]
  // CHECK: %[[TMP_98:.*]] = mhlo.power %[[TMP_97]], %[[TMP_91]]
  // CHECK: %[[TMP_99:.*]] = mhlo.add %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.add %[[TMP_97]], %[[TMP_93]]
  // CHECK: %[[TMP_101:.*]] = mhlo.power %[[TMP_100]], %[[TMP_91]]
  // CHECK: %[[TMP_102:.*]] = mhlo.add %[[TMP_99]], %[[TMP_101]]
  // CHECK: %[[TMP_103:.*]] = mhlo.add %[[TMP_100]], %[[TMP_93]]
  // CHECK: %[[TMP_104:.*]] = mhlo.power %[[TMP_103]], %[[TMP_91]]
  // CHECK: %[[TMP_105:.*]] = mhlo.add %[[TMP_102]], %[[TMP_104]]
  // CHECK: %[[TMP_106:.*]] = mhlo.add %[[TMP_103]], %[[TMP_93]]
  // CHECK: %[[TMP_107:.*]] = mhlo.power %[[TMP_106]], %[[TMP_91]]
  // CHECK: %[[TMP_108:.*]] = mhlo.add %[[TMP_105]], %[[TMP_107]]
  // CHECK: %[[TMP_109:.*]] = mhlo.add %[[TMP_106]], %[[TMP_93]]
  // CHECK: %[[TMP_110:.*]] = mhlo.power %[[TMP_109]], %[[TMP_91]]
  // CHECK: %[[TMP_111:.*]] = mhlo.add %[[TMP_108]], %[[TMP_110]]
  // CHECK: %[[TMP_112:.*]] = mhlo.add %[[TMP_109]], %[[TMP_93]]
  // CHECK: %[[TMP_113:.*]] = mhlo.power %[[TMP_112]], %[[TMP_91]]
  // CHECK: %[[TMP_114:.*]] = mhlo.add %[[TMP_111]], %[[TMP_113]]
  // CHECK: %[[TMP_115:.*]] = mhlo.add %[[TMP_112]], %[[TMP_93]]
  // CHECK: %[[TMP_116:.*]] = mhlo.power %[[TMP_115]], %[[TMP_91]]
  // CHECK: %[[TMP_117:.*]] = mhlo.add %[[TMP_114]], %[[TMP_116]]
  // CHECK: %[[TMP_118:.*]] = mhlo.add %[[TMP_115]], %[[TMP_93]]
  // CHECK: %[[TMP_119:.*]] = mhlo.power %[[TMP_118]], %[[TMP_91]]
  // CHECK: %[[TMP_120:.*]] = mhlo.add %[[TMP_117]], %[[TMP_119]]
  // CHECK: %[[TMP_121:.*]] = mhlo.add %[[TMP_118]], %[[TMP_93]]
  // CHECK: %[[TMP_122:.*]] = mhlo.power %[[TMP_121]], %[[TMP_91]]
  // CHECK: %[[TMP_123:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_124:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_123]]
  // CHECK: %[[TMP_125:.*]] = mhlo.multiply %[[TMP_122]], %[[TMP_121]]
  // CHECK: %[[TMP_126:.*]] = mhlo.divide %[[TMP_125]], %[[TMP_124]]
  // CHECK: %[[TMP_127:.*]] = mhlo.add %[[TMP_120]], %[[TMP_126]]
  // CHECK: %[[TMP_128:.*]] = mhlo.multiply %[[TMP_121]], %[[TMP_121]]
  // CHECK: %[[TMP_129:.*]] = mhlo.divide %[[TMP_93]], %[[TMP_128]]
  // CHECK: %[[TMP_130:.*]] = mhlo.constant dense<2.200000e+01>
  // CHECK: %[[TMP_131:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_130]]
  // CHECK: %[[TMP_132:.*]] = mhlo.constant dense<2.100000e+01>
  // CHECK: %[[TMP_133:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_132]]
  // CHECK: %[[TMP_134:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_133]]
  // CHECK: %[[TMP_135:.*]] = mhlo.constant dense<-1.39544646E-19>
  // CHECK: %[[TMP_136:.*]] = mhlo.add %[[TMP_90]], %[[TMP_135]]
  // CHECK: %[[TMP_137:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_136]]
  // CHECK: %[[TMP_138:.*]] = mhlo.multiply %[[TMP_134]], %[[TMP_137]]
  // CHECK: %[[TMP_139:.*]] = mhlo.constant dense<2.000000e+01>
  // CHECK: %[[TMP_140:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_139]]
  // CHECK: %[[TMP_141:.*]] = mhlo.constant dense<1.900000e+01>
  // CHECK: %[[TMP_142:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_141]]
  // CHECK: %[[TMP_143:.*]] = mhlo.multiply %[[TMP_140]], %[[TMP_142]]
  // CHECK: %[[TMP_144:.*]] = mhlo.constant dense<5.50900303E-18>
  // CHECK: %[[TMP_145:.*]] = mhlo.add %[[TMP_138]], %[[TMP_144]]
  // CHECK: %[[TMP_146:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_145]]
  // CHECK: %[[TMP_147:.*]] = mhlo.multiply %[[TMP_143]], %[[TMP_146]]
  // CHECK: %[[TMP_148:.*]] = mhlo.constant dense<1.800000e+01>
  // CHECK: %[[TMP_149:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_148]]
  // CHECK: %[[TMP_150:.*]] = mhlo.constant dense<1.700000e+01>
  // CHECK: %[[TMP_151:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_150]]
  // CHECK: %[[TMP_152:.*]] = mhlo.multiply %[[TMP_149]], %[[TMP_151]]
  // CHECK: %[[TMP_153:.*]] = mhlo.constant dense<-2.17486866E-16>
  // CHECK: %[[TMP_154:.*]] = mhlo.add %[[TMP_147]], %[[TMP_153]]
  // CHECK: %[[TMP_155:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_154]]
  // CHECK: %[[TMP_156:.*]] = mhlo.multiply %[[TMP_152]], %[[TMP_155]]
  // CHECK: %[[TMP_157:.*]] = mhlo.constant dense<1.600000e+01>
  // CHECK: %[[TMP_158:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_157]]
  // CHECK: %[[TMP_159:.*]] = mhlo.constant dense<1.500000e+01>
  // CHECK: %[[TMP_160:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_159]]
  // CHECK: %[[TMP_161:.*]] = mhlo.multiply %[[TMP_158]], %[[TMP_160]]
  // CHECK: %[[TMP_162:.*]] = mhlo.constant dense<8.58606213E-15>
  // CHECK: %[[TMP_163:.*]] = mhlo.add %[[TMP_156]], %[[TMP_162]]
  // CHECK: %[[TMP_164:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_163]]
  // CHECK: %[[TMP_165:.*]] = mhlo.multiply %[[TMP_161]], %[[TMP_164]]
  // CHECK: %[[TMP_166:.*]] = mhlo.constant dense<1.400000e+01>
  // CHECK: %[[TMP_167:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_166]]
  // CHECK: %[[TMP_168:.*]] = mhlo.constant dense<1.300000e+01>
  // CHECK: %[[TMP_169:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_168]]
  // CHECK: %[[TMP_170:.*]] = mhlo.multiply %[[TMP_167]], %[[TMP_169]]
  // CHECK: %[[TMP_171:.*]] = mhlo.constant dense<-3.3896803E-13>
  // CHECK: %[[TMP_172:.*]] = mhlo.add %[[TMP_165]], %[[TMP_171]]
  // CHECK: %[[TMP_173:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_172]]
  // CHECK: %[[TMP_174:.*]] = mhlo.multiply %[[TMP_170]], %[[TMP_173]]
  // CHECK: %[[TMP_175:.*]] = mhlo.constant dense<1.200000e+01>
  // CHECK: %[[TMP_176:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_175]]
  // CHECK: %[[TMP_177:.*]] = mhlo.constant dense<1.100000e+01>
  // CHECK: %[[TMP_178:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_177]]
  // CHECK: %[[TMP_179:.*]] = mhlo.multiply %[[TMP_176]], %[[TMP_178]]
  // CHECK: %[[TMP_180:.*]] = mhlo.constant dense<1.33825364E-11>
  // CHECK: %[[TMP_181:.*]] = mhlo.add %[[TMP_174]], %[[TMP_180]]
  // CHECK: %[[TMP_182:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_181]]
  // CHECK: %[[TMP_183:.*]] = mhlo.multiply %[[TMP_179]], %[[TMP_182]]
  // CHECK: %[[TMP_184:.*]] = mhlo.constant dense<1.000000e+01>
  // CHECK: %[[TMP_185:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_184]]
  // CHECK: %[[TMP_186:.*]] = mhlo.constant dense<9.000000e+00>
  // CHECK: %[[TMP_187:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_186]]
  // CHECK: %[[TMP_188:.*]] = mhlo.multiply %[[TMP_185]], %[[TMP_187]]
  // CHECK: %[[TMP_189:.*]] = mhlo.constant dense<-5.28419031E-10>
  // CHECK: %[[TMP_190:.*]] = mhlo.add %[[TMP_183]], %[[TMP_189]]
  // CHECK: %[[TMP_191:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_190]]
  // CHECK: %[[TMP_192:.*]] = mhlo.multiply %[[TMP_188]], %[[TMP_191]]
  // CHECK: %[[TMP_193:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_194:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_193]]
  // CHECK: %[[TMP_195:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_196:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_195]]
  // CHECK: %[[TMP_197:.*]] = mhlo.multiply %[[TMP_194]], %[[TMP_196]]
  // CHECK: %[[TMP_198:.*]] = mhlo.constant dense<2.08767563E-8>
  // CHECK: %[[TMP_199:.*]] = mhlo.add %[[TMP_192]], %[[TMP_198]]
  // CHECK: %[[TMP_200:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_199]]
  // CHECK: %[[TMP_201:.*]] = mhlo.multiply %[[TMP_197]], %[[TMP_200]]
  // CHECK: %[[TMP_202:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_203:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_202]]
  // CHECK: %[[TMP_204:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_205:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_204]]
  // CHECK: %[[TMP_206:.*]] = mhlo.multiply %[[TMP_203]], %[[TMP_205]]
  // CHECK: %[[TMP_207:.*]] = mhlo.constant dense<-8.26719599E-7>
  // CHECK: %[[TMP_208:.*]] = mhlo.add %[[TMP_201]], %[[TMP_207]]
  // CHECK: %[[TMP_209:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_208]]
  // CHECK: %[[TMP_210:.*]] = mhlo.multiply %[[TMP_206]], %[[TMP_209]]
  // CHECK: %[[TMP_211:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_212:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_211]]
  // CHECK: %[[TMP_213:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_214:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_213]]
  // CHECK: %[[TMP_215:.*]] = mhlo.multiply %[[TMP_212]], %[[TMP_214]]
  // CHECK: %[[TMP_216:.*]] = mhlo.constant dense<3.30687835E-5>
  // CHECK: %[[TMP_217:.*]] = mhlo.add %[[TMP_210]], %[[TMP_216]]
  // CHECK: %[[TMP_218:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_217]]
  // CHECK: %[[TMP_219:.*]] = mhlo.multiply %[[TMP_215]], %[[TMP_218]]
  // CHECK: %[[TMP_220:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_221:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_220]]
  // CHECK: %[[TMP_222:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_223:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_222]]
  // CHECK: %[[TMP_224:.*]] = mhlo.multiply %[[TMP_221]], %[[TMP_223]]
  // CHECK: %[[TMP_225:.*]] = mhlo.constant dense<-0.00138888892>
  // CHECK: %[[TMP_226:.*]] = mhlo.add %[[TMP_219]], %[[TMP_225]]
  // CHECK: %[[TMP_227:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_226]]
  // CHECK: %[[TMP_228:.*]] = mhlo.multiply %[[TMP_224]], %[[TMP_227]]
  // CHECK: %[[TMP_229:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_230:.*]] = mhlo.divide %[[TMP_5]], %[[TMP_121]]
  // CHECK: %[[TMP_231:.*]] = mhlo.constant dense<0.0833333358>
  // CHECK: %[[TMP_232:.*]] = mhlo.add %[[TMP_231]], %[[TMP_228]]
  // CHECK: %[[TMP_233:.*]] = mhlo.multiply %[[TMP_230]], %[[TMP_232]]
  // CHECK: %[[TMP_234:.*]] = mhlo.add %[[TMP_229]], %[[TMP_233]]
  // CHECK: %[[TMP_235:.*]] = mhlo.multiply %[[TMP_122]], %[[TMP_234]]
  // CHECK: %[[TMP_236:.*]] = mhlo.add %[[TMP_127]], %[[TMP_235]]
  // CHECK: %[[TMP_237:.*]] = "mhlo.abs"(%[[TMP_122]])
  // CHECK: %[[TMP_238:.*]] = "mhlo.abs"(%[[TMP_120]])
  // CHECK: %[[TMP_239:.*]] = mhlo.constant dense<1.401300e-45>
  // CHECK: %[[TMP_240:.*]] = mhlo.multiply %[[TMP_238]], %[[TMP_239]]
  // CHECK: %[[TMP_241:.*]] = "mhlo.compare"(%[[TMP_237]], %[[TMP_240]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_242:.*]] = "mhlo.select"(%[[TMP_241]], %[[TMP_120]], %[[TMP_236]])
  // CHECK: %[[TMP_243:.*]] = mhlo.constant dense<0x7FC00000>
  // CHECK: %[[TMP_244:.*]] = "mhlo.compare"(%[[TMP_5]], %[[TMP_123]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_245:.*]] = "mhlo.select"(%[[TMP_244]], %[[TMP_243]], %[[TMP_242]])
  // CHECK: %[[TMP_246:.*]] = "mhlo.compare"(%[[ARG1]], %[[TMP_90]]) {comparison_direction = "LE"}
  // CHECK: %[[TMP_247:.*]] = "mhlo.floor"(%[[TMP_5]])
  // CHECK: %[[TMP_248:.*]] = "mhlo.compare"(%[[TMP_5]], %[[TMP_247]]) {comparison_direction = "NE"}
  // CHECK: %[[TMP_249:.*]] = mhlo.and %[[TMP_246]], %[[TMP_248]]
  // CHECK: %[[TMP_250:.*]] = "mhlo.select"(%[[TMP_249]], %[[TMP_243]], %[[TMP_245]])
  // CHECK: %[[TMP_251:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_252:.*]] = "mhlo.floor"(%[[ARG1]])
  // CHECK: %[[TMP_253:.*]] = "mhlo.compare"(%[[ARG1]], %[[TMP_252]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_254:.*]] = mhlo.and %[[TMP_246]], %[[TMP_253]]
  // CHECK: %[[TMP_255:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_256:.*]] = "mhlo.floor"(%[[TMP_5]])
  // CHECK: %[[TMP_257:.*]] = "mhlo.compare"(%[[TMP_5]], %[[TMP_256]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_258:.*]] = mhlo.remainder %[[TMP_5]], %[[TMP_255]]
  // CHECK: %[[TMP_259:.*]] = "mhlo.compare"(%[[TMP_258]], %[[TMP_90]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_260:.*]] = mhlo.and %[[TMP_257]], %[[TMP_259]]
  // CHECK: %[[TMP_261:.*]] = "mhlo.select"(%[[TMP_260]], %[[TMP_251]], %[[TMP_243]])
  // CHECK: %[[TMP_262:.*]] = "mhlo.select"(%[[TMP_254]], %[[TMP_261]], %[[TMP_250]])
  // CHECK: %[[TMP_263:.*]] = "mhlo.compare"(%[[TMP_5]], %[[TMP_93]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_264:.*]] = "mhlo.select"(%[[TMP_263]], %[[TMP_251]], %[[TMP_262]])
  // CHECK: %[[TMP_265:.*]] = mhlo.multiply %[[TMP_4]], %[[TMP_89]]
  // CHECK: %[[TMP_266:.*]] = mhlo.multiply %[[TMP_265]], %[[TMP_264]]
  // CHECK: %[[TMP_267:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_268:.*]] = "mhlo.compare"(%[[ARG0]], %[[TMP_267]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_269:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_270:.*]] = "mhlo.compare"(%[[ARG1]], %[[TMP_269]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_271:.*]] = "mhlo.negate"(%[[ARG1]])
  // CHECK: %[[TMP_272:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_273:.*]] = mhlo.subtract %[[ARG1]], %[[TMP_272]]
  // CHECK: %[[TMP_274:.*]] = "mhlo.select"(%[[TMP_270]], %[[TMP_271]], %[[TMP_273]])
  // CHECK: %[[TMP_275:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_276:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_277:.*]] = mhlo.constant dense<676.520386>
  // CHECK: %[[TMP_278:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_279:.*]] = mhlo.add %[[TMP_274]], %[[TMP_278]]
  // CHECK: %[[TMP_280:.*]] = mhlo.multiply %[[TMP_279]], %[[TMP_279]]
  // CHECK: %[[TMP_281:.*]] = mhlo.divide %[[TMP_277]], %[[TMP_280]]
  // CHECK: %[[TMP_282:.*]] = mhlo.subtract %[[TMP_275]], %[[TMP_281]]
  // CHECK: %[[TMP_283:.*]] = mhlo.divide %[[TMP_277]], %[[TMP_279]]
  // CHECK: %[[TMP_284:.*]] = mhlo.add %[[TMP_276]], %[[TMP_283]]
  // CHECK: %[[TMP_285:.*]] = mhlo.constant dense<-1259.13916>
  // CHECK: %[[TMP_286:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_287:.*]] = mhlo.add %[[TMP_274]], %[[TMP_286]]
  // CHECK: %[[TMP_288:.*]] = mhlo.multiply %[[TMP_287]], %[[TMP_287]]
  // CHECK: %[[TMP_289:.*]] = mhlo.divide %[[TMP_285]], %[[TMP_288]]
  // CHECK: %[[TMP_290:.*]] = mhlo.subtract %[[TMP_282]], %[[TMP_289]]
  // CHECK: %[[TMP_291:.*]] = mhlo.divide %[[TMP_285]], %[[TMP_287]]
  // CHECK: %[[TMP_292:.*]] = mhlo.add %[[TMP_284]], %[[TMP_291]]
  // CHECK: %[[TMP_293:.*]] = mhlo.constant dense<771.323425>
  // CHECK: %[[TMP_294:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_295:.*]] = mhlo.add %[[TMP_274]], %[[TMP_294]]
  // CHECK: %[[TMP_296:.*]] = mhlo.multiply %[[TMP_295]], %[[TMP_295]]
  // CHECK: %[[TMP_297:.*]] = mhlo.divide %[[TMP_293]], %[[TMP_296]]
  // CHECK: %[[TMP_298:.*]] = mhlo.subtract %[[TMP_290]], %[[TMP_297]]
  // CHECK: %[[TMP_299:.*]] = mhlo.divide %[[TMP_293]], %[[TMP_295]]
  // CHECK: %[[TMP_300:.*]] = mhlo.add %[[TMP_292]], %[[TMP_299]]
  // CHECK: %[[TMP_301:.*]] = mhlo.constant dense<-176.615036>
  // CHECK: %[[TMP_302:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_303:.*]] = mhlo.add %[[TMP_274]], %[[TMP_302]]
  // CHECK: %[[TMP_304:.*]] = mhlo.multiply %[[TMP_303]], %[[TMP_303]]
  // CHECK: %[[TMP_305:.*]] = mhlo.divide %[[TMP_301]], %[[TMP_304]]
  // CHECK: %[[TMP_306:.*]] = mhlo.subtract %[[TMP_298]], %[[TMP_305]]
  // CHECK: %[[TMP_307:.*]] = mhlo.divide %[[TMP_301]], %[[TMP_303]]
  // CHECK: %[[TMP_308:.*]] = mhlo.add %[[TMP_300]], %[[TMP_307]]
  // CHECK: %[[TMP_309:.*]] = mhlo.constant dense<12.5073433>
  // CHECK: %[[TMP_310:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_311:.*]] = mhlo.add %[[TMP_274]], %[[TMP_310]]
  // CHECK: %[[TMP_312:.*]] = mhlo.multiply %[[TMP_311]], %[[TMP_311]]
  // CHECK: %[[TMP_313:.*]] = mhlo.divide %[[TMP_309]], %[[TMP_312]]
  // CHECK: %[[TMP_314:.*]] = mhlo.subtract %[[TMP_306]], %[[TMP_313]]
  // CHECK: %[[TMP_315:.*]] = mhlo.divide %[[TMP_309]], %[[TMP_311]]
  // CHECK: %[[TMP_316:.*]] = mhlo.add %[[TMP_308]], %[[TMP_315]]
  // CHECK: %[[TMP_317:.*]] = mhlo.constant dense<-0.138571098>
  // CHECK: %[[TMP_318:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_319:.*]] = mhlo.add %[[TMP_274]], %[[TMP_318]]
  // CHECK: %[[TMP_320:.*]] = mhlo.multiply %[[TMP_319]], %[[TMP_319]]
  // CHECK: %[[TMP_321:.*]] = mhlo.divide %[[TMP_317]], %[[TMP_320]]
  // CHECK: %[[TMP_322:.*]] = mhlo.subtract %[[TMP_314]], %[[TMP_321]]
  // CHECK: %[[TMP_323:.*]] = mhlo.divide %[[TMP_317]], %[[TMP_319]]
  // CHECK: %[[TMP_324:.*]] = mhlo.add %[[TMP_316]], %[[TMP_323]]
  // CHECK: %[[TMP_325:.*]] = mhlo.constant dense<9.98436917E-6>
  // CHECK: %[[TMP_326:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_327:.*]] = mhlo.add %[[TMP_274]], %[[TMP_326]]
  // CHECK: %[[TMP_328:.*]] = mhlo.multiply %[[TMP_327]], %[[TMP_327]]
  // CHECK: %[[TMP_329:.*]] = mhlo.divide %[[TMP_325]], %[[TMP_328]]
  // CHECK: %[[TMP_330:.*]] = mhlo.subtract %[[TMP_322]], %[[TMP_329]]
  // CHECK: %[[TMP_331:.*]] = mhlo.divide %[[TMP_325]], %[[TMP_327]]
  // CHECK: %[[TMP_332:.*]] = mhlo.add %[[TMP_324]], %[[TMP_331]]
  // CHECK: %[[TMP_333:.*]] = mhlo.constant dense<1.50563267E-7>
  // CHECK: %[[TMP_334:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_335:.*]] = mhlo.add %[[TMP_274]], %[[TMP_334]]
  // CHECK: %[[TMP_336:.*]] = mhlo.multiply %[[TMP_335]], %[[TMP_335]]
  // CHECK: %[[TMP_337:.*]] = mhlo.divide %[[TMP_333]], %[[TMP_336]]
  // CHECK: %[[TMP_338:.*]] = mhlo.subtract %[[TMP_330]], %[[TMP_337]]
  // CHECK: %[[TMP_339:.*]] = mhlo.divide %[[TMP_333]], %[[TMP_335]]
  // CHECK: %[[TMP_340:.*]] = mhlo.add %[[TMP_332]], %[[TMP_339]]
  // CHECK: %[[TMP_341:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_342:.*]] = mhlo.add %[[TMP_341]], %[[TMP_274]]
  // CHECK: %[[TMP_343:.*]] = mhlo.constant dense<2.01490307>
  // CHECK: %[[TMP_344:.*]] = mhlo.divide %[[TMP_274]], %[[TMP_341]]
  // CHECK: %[[TMP_345:.*]] = "mhlo.log_plus_one"(%[[TMP_344]])
  // CHECK: %[[TMP_346:.*]] = mhlo.add %[[TMP_343]], %[[TMP_345]]
  // CHECK: %[[TMP_347:.*]] = mhlo.divide %[[TMP_338]], %[[TMP_340]]
  // CHECK: %[[TMP_348:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_349:.*]] = mhlo.divide %[[TMP_348]], %[[TMP_342]]
  // CHECK: %[[TMP_350:.*]] = mhlo.add %[[TMP_346]], %[[TMP_347]]
  // CHECK: %[[TMP_351:.*]] = mhlo.subtract %[[TMP_350]], %[[TMP_349]]
  // CHECK: %[[TMP_352:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_353:.*]] = mhlo.add %[[ARG1]], %[[TMP_352]]
  // CHECK: %[[TMP_354:.*]] = "mhlo.floor"(%[[TMP_353]])
  // CHECK: %[[TMP_355:.*]] = "mhlo.abs"(%[[TMP_354]])
  // CHECK: %[[TMP_356:.*]] = mhlo.add %[[ARG1]], %[[TMP_355]]
  // CHECK: %[[TMP_357:.*]] = mhlo.constant dense<3.14159274>
  // CHECK: %[[TMP_358:.*]] = mhlo.multiply %[[TMP_357]], %[[TMP_356]]
  // CHECK: %[[TMP_359:.*]] = "mhlo.cosine"(%[[TMP_358]])
  // CHECK: %[[TMP_360:.*]] = "mhlo.sine"(%[[TMP_358]])
  // CHECK: %[[TMP_361:.*]] = mhlo.multiply %[[TMP_357]], %[[TMP_359]]
  // CHECK: %[[TMP_362:.*]] = mhlo.divide %[[TMP_361]], %[[TMP_360]]
  // CHECK: %[[TMP_363:.*]] = mhlo.subtract %[[TMP_351]], %[[TMP_362]]
  // CHECK: %[[TMP_364:.*]] = "mhlo.select"(%[[TMP_270]], %[[TMP_363]], %[[TMP_351]])
  // CHECK: %[[TMP_365:.*]] = "mhlo.compare"(%[[ARG1]], %[[TMP_275]]) {comparison_direction = "LE"}
  // CHECK: %[[TMP_366:.*]] = "mhlo.floor"(%[[ARG1]])
  // CHECK: %[[TMP_367:.*]] = "mhlo.compare"(%[[ARG1]], %[[TMP_366]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_368:.*]] = mhlo.and %[[TMP_365]], %[[TMP_367]]
  // CHECK: %[[TMP_369:.*]] = mhlo.constant dense<0x7FC00000>
  // CHECK: %[[TMP_370:.*]] = "mhlo.select"(%[[TMP_368]], %[[TMP_369]], %[[TMP_364]])
  // CHECK: %[[TMP_371:.*]] = "mhlo.select"(%[[TMP_268]], %[[TMP_370]], %[[TMP_266]])
  // CHECK: %[[TMP_372:.*]] = "mhlo.floor"(%[[ARG0]])
  // CHECK: %[[TMP_373:.*]] = "mhlo.compare"(%[[ARG0]], %[[TMP_372]]) {comparison_direction = "NE"}
  // CHECK: %[[TMP_374:.*]] = "mhlo.compare"(%[[ARG0]], %[[TMP_267]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_375:.*]] = mhlo.or %[[TMP_373]], %[[TMP_374]]
  // CHECK: %[[TMP_376:.*]] = mhlo.constant dense<0x7FC00000>
  // CHECK: %[[TMP_377:.*]] = "mhlo.select"(%[[TMP_375]], %[[TMP_376]], %[[TMP_371]])
  %1 = chlo.polygamma %lhs, %rhs : tensor<f32>, tensor<f32> -> tensor<f32>
  return %1 : tensor<f32>
}

// ----

// CHECK: @polygamma_f64
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f64>, %[[ARG1:.*]]: tensor<f64>)
func @polygamma_f64(%lhs : tensor<f64>, %rhs : tensor<f64>) -> tensor<f64> {
  // CHECK: %[[TMP_0:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_1:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_2:.*]] = mhlo.remainder %[[ARG0]], %[[TMP_1]]
  // CHECK: %[[TMP_3:.*]] = mhlo.multiply %[[TMP_1]], %[[TMP_2]]
  // CHECK: %[[TMP_4:.*]] = mhlo.subtract %[[TMP_3]], %[[TMP_0]]
  // CHECK: %[[TMP_5:.*]] = mhlo.add %[[ARG0]], %[[TMP_0]]
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_7:.*]] = "mhlo.compare"(%[[TMP_5]], %[[TMP_6]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_8:.*]] = "mhlo.negate"(%[[TMP_5]])
  // CHECK: %[[TMP_9:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = "mhlo.select"(%[[TMP_7]], %[[TMP_8]], %[[TMP_10]])
  // CHECK: %[[TMP_12:.*]] = mhlo.constant dense<0.99999999999980993>
  // CHECK: %[[TMP_13:.*]] = mhlo.constant dense<676.5203681218851>
  // CHECK: %[[TMP_14:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_11]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = mhlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = mhlo.add %[[TMP_12]], %[[TMP_16]]
  // CHECK: %[[TMP_18:.*]] = mhlo.constant dense<-1259.1392167224028>
  // CHECK: %[[TMP_19:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_11]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK: %[[TMP_23:.*]] = mhlo.constant dense<771.32342877765313>
  // CHECK: %[[TMP_24:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = mhlo.add %[[TMP_11]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = mhlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = mhlo.constant dense<-176.61502916214059>
  // CHECK: %[[TMP_29:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = mhlo.add %[[TMP_11]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = mhlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK: %[[TMP_33:.*]] = mhlo.constant dense<12.507343278686905>
  // CHECK: %[[TMP_34:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = mhlo.add %[[TMP_11]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = mhlo.constant dense<-0.13857109526572012>
  // CHECK: %[[TMP_39:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = mhlo.add %[[TMP_11]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = mhlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = mhlo.constant dense<9.9843695780195716E-6>
  // CHECK: %[[TMP_44:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = mhlo.add %[[TMP_11]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK: %[[TMP_48:.*]] = mhlo.constant dense<1.5056327351493116E-7>
  // CHECK: %[[TMP_49:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_11]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_54:.*]] = mhlo.add %[[TMP_53]], %[[TMP_11]]
  // CHECK: %[[TMP_55:.*]] = mhlo.constant dense<2.0149030205422647>
  // CHECK: %[[TMP_56:.*]] = mhlo.divide %[[TMP_11]], %[[TMP_53]]
  // CHECK: %[[TMP_57:.*]] = "mhlo.log_plus_one"(%[[TMP_56]])
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_55]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = mhlo.divide %[[TMP_54]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = mhlo.add %[[TMP_11]], %[[TMP_6]]
  // CHECK: %[[TMP_61:.*]] = mhlo.subtract %[[TMP_60]], %[[TMP_59]]
  // CHECK: %[[TMP_62:.*]] = mhlo.multiply %[[TMP_61]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = "mhlo.log"(%[[TMP_52]])
  // CHECK: %[[TMP_64:.*]] = mhlo.constant dense<0.91893853320467266>
  // CHECK: %[[TMP_65:.*]] = mhlo.add %[[TMP_64]], %[[TMP_62]]
  // CHECK: %[[TMP_66:.*]] = mhlo.add %[[TMP_65]], %[[TMP_63]]
  // CHECK: %[[TMP_67:.*]] = "mhlo.abs"(%[[TMP_5]])
  // CHECK: %[[TMP_68:.*]] = "mhlo.floor"(%[[TMP_67]])
  // CHECK: %[[TMP_69:.*]] = mhlo.subtract %[[TMP_67]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = "mhlo.compare"(%[[TMP_6]], %[[TMP_69]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_71:.*]] = mhlo.subtract %[[TMP_9]], %[[TMP_69]]
  // CHECK: %[[TMP_72:.*]] = "mhlo.select"(%[[TMP_70]], %[[TMP_71]], %[[TMP_69]])
  // CHECK: %[[TMP_73:.*]] = mhlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_74:.*]] = mhlo.multiply %[[TMP_73]], %[[TMP_72]]
  // CHECK: %[[TMP_75:.*]] = "mhlo.sine"(%[[TMP_74]])
  // CHECK: %[[TMP_76:.*]] = "mhlo.log"(%[[TMP_75]])
  // CHECK: %[[TMP_77:.*]] = mhlo.constant dense<1.1447298858494002>
  // CHECK: %[[TMP_78:.*]] = mhlo.subtract %[[TMP_77]], %[[TMP_76]]
  // CHECK: %[[TMP_79:.*]] = mhlo.subtract %[[TMP_78]], %[[TMP_66]]
  // CHECK: %[[TMP_80:.*]] = "mhlo.is_finite"(%[[TMP_76]])
  // CHECK: %[[TMP_81:.*]] = "mhlo.negate"(%[[TMP_76]])
  // CHECK: %[[TMP_82:.*]] = "mhlo.select"(%[[TMP_80]], %[[TMP_79]], %[[TMP_81]])
  // CHECK: %[[TMP_83:.*]] = "mhlo.select"(%[[TMP_7]], %[[TMP_82]], %[[TMP_66]])
  // CHECK: %[[TMP_84:.*]] = "mhlo.abs"(%[[TMP_5]])
  // CHECK: %[[TMP_85:.*]] = mhlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_86:.*]] = "mhlo.compare"(%[[TMP_84]], %[[TMP_85]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_87:.*]] = mhlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_88:.*]] = "mhlo.select"(%[[TMP_86]], %[[TMP_87]], %[[TMP_83]])
  // CHECK: %[[TMP_89:.*]] = "mhlo.exponential"(%[[TMP_88]])
  // CHECK: %[[TMP_90:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_91:.*]] = "mhlo.negate"(%[[TMP_5]])
  // CHECK: %[[TMP_92:.*]] = mhlo.power %[[ARG1]], %[[TMP_91]]
  // CHECK: %[[TMP_93:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_94:.*]] = mhlo.add %[[ARG1]], %[[TMP_93]]
  // CHECK: %[[TMP_95:.*]] = mhlo.power %[[TMP_94]], %[[TMP_91]]
  // CHECK: %[[TMP_96:.*]] = mhlo.add %[[TMP_92]], %[[TMP_95]]
  // CHECK: %[[TMP_97:.*]] = mhlo.add %[[TMP_94]], %[[TMP_93]]
  // CHECK: %[[TMP_98:.*]] = mhlo.power %[[TMP_97]], %[[TMP_91]]
  // CHECK: %[[TMP_99:.*]] = mhlo.add %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.add %[[TMP_97]], %[[TMP_93]]
  // CHECK: %[[TMP_101:.*]] = mhlo.power %[[TMP_100]], %[[TMP_91]]
  // CHECK: %[[TMP_102:.*]] = mhlo.add %[[TMP_99]], %[[TMP_101]]
  // CHECK: %[[TMP_103:.*]] = mhlo.add %[[TMP_100]], %[[TMP_93]]
  // CHECK: %[[TMP_104:.*]] = mhlo.power %[[TMP_103]], %[[TMP_91]]
  // CHECK: %[[TMP_105:.*]] = mhlo.add %[[TMP_102]], %[[TMP_104]]
  // CHECK: %[[TMP_106:.*]] = mhlo.add %[[TMP_103]], %[[TMP_93]]
  // CHECK: %[[TMP_107:.*]] = mhlo.power %[[TMP_106]], %[[TMP_91]]
  // CHECK: %[[TMP_108:.*]] = mhlo.add %[[TMP_105]], %[[TMP_107]]
  // CHECK: %[[TMP_109:.*]] = mhlo.add %[[TMP_106]], %[[TMP_93]]
  // CHECK: %[[TMP_110:.*]] = mhlo.power %[[TMP_109]], %[[TMP_91]]
  // CHECK: %[[TMP_111:.*]] = mhlo.add %[[TMP_108]], %[[TMP_110]]
  // CHECK: %[[TMP_112:.*]] = mhlo.add %[[TMP_109]], %[[TMP_93]]
  // CHECK: %[[TMP_113:.*]] = mhlo.power %[[TMP_112]], %[[TMP_91]]
  // CHECK: %[[TMP_114:.*]] = mhlo.add %[[TMP_111]], %[[TMP_113]]
  // CHECK: %[[TMP_115:.*]] = mhlo.add %[[TMP_112]], %[[TMP_93]]
  // CHECK: %[[TMP_116:.*]] = mhlo.power %[[TMP_115]], %[[TMP_91]]
  // CHECK: %[[TMP_117:.*]] = mhlo.add %[[TMP_114]], %[[TMP_116]]
  // CHECK: %[[TMP_118:.*]] = mhlo.add %[[TMP_115]], %[[TMP_93]]
  // CHECK: %[[TMP_119:.*]] = mhlo.power %[[TMP_118]], %[[TMP_91]]
  // CHECK: %[[TMP_120:.*]] = mhlo.add %[[TMP_117]], %[[TMP_119]]
  // CHECK: %[[TMP_121:.*]] = mhlo.add %[[TMP_118]], %[[TMP_93]]
  // CHECK: %[[TMP_122:.*]] = mhlo.power %[[TMP_121]], %[[TMP_91]]
  // CHECK: %[[TMP_123:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_124:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_123]]
  // CHECK: %[[TMP_125:.*]] = mhlo.multiply %[[TMP_122]], %[[TMP_121]]
  // CHECK: %[[TMP_126:.*]] = mhlo.divide %[[TMP_125]], %[[TMP_124]]
  // CHECK: %[[TMP_127:.*]] = mhlo.add %[[TMP_120]], %[[TMP_126]]
  // CHECK: %[[TMP_128:.*]] = mhlo.multiply %[[TMP_121]], %[[TMP_121]]
  // CHECK: %[[TMP_129:.*]] = mhlo.divide %[[TMP_93]], %[[TMP_128]]
  // CHECK: %[[TMP_130:.*]] = mhlo.constant dense<2.200000e+01>
  // CHECK: %[[TMP_131:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_130]]
  // CHECK: %[[TMP_132:.*]] = mhlo.constant dense<2.100000e+01>
  // CHECK: %[[TMP_133:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_132]]
  // CHECK: %[[TMP_134:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_133]]
  // CHECK: %[[TMP_135:.*]] = mhlo.constant dense<-1.3954464685812522E-19>
  // CHECK: %[[TMP_136:.*]] = mhlo.add %[[TMP_90]], %[[TMP_135]]
  // CHECK: %[[TMP_137:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_136]]
  // CHECK: %[[TMP_138:.*]] = mhlo.multiply %[[TMP_134]], %[[TMP_137]]
  // CHECK: %[[TMP_139:.*]] = mhlo.constant dense<2.000000e+01>
  // CHECK: %[[TMP_140:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_139]]
  // CHECK: %[[TMP_141:.*]] = mhlo.constant dense<1.900000e+01>
  // CHECK: %[[TMP_142:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_141]]
  // CHECK: %[[TMP_143:.*]] = mhlo.multiply %[[TMP_140]], %[[TMP_142]]
  // CHECK: %[[TMP_144:.*]] = mhlo.constant dense<5.5090028283602295E-18>
  // CHECK: %[[TMP_145:.*]] = mhlo.add %[[TMP_138]], %[[TMP_144]]
  // CHECK: %[[TMP_146:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_145]]
  // CHECK: %[[TMP_147:.*]] = mhlo.multiply %[[TMP_143]], %[[TMP_146]]
  // CHECK: %[[TMP_148:.*]] = mhlo.constant dense<1.800000e+01>
  // CHECK: %[[TMP_149:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_148]]
  // CHECK: %[[TMP_150:.*]] = mhlo.constant dense<1.700000e+01>
  // CHECK: %[[TMP_151:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_150]]
  // CHECK: %[[TMP_152:.*]] = mhlo.multiply %[[TMP_149]], %[[TMP_151]]
  // CHECK: %[[TMP_153:.*]] = mhlo.constant dense<-2.1748686985580617E-16>
  // CHECK: %[[TMP_154:.*]] = mhlo.add %[[TMP_147]], %[[TMP_153]]
  // CHECK: %[[TMP_155:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_154]]
  // CHECK: %[[TMP_156:.*]] = mhlo.multiply %[[TMP_152]], %[[TMP_155]]
  // CHECK: %[[TMP_157:.*]] = mhlo.constant dense<1.600000e+01>
  // CHECK: %[[TMP_158:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_157]]
  // CHECK: %[[TMP_159:.*]] = mhlo.constant dense<1.500000e+01>
  // CHECK: %[[TMP_160:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_159]]
  // CHECK: %[[TMP_161:.*]] = mhlo.multiply %[[TMP_158]], %[[TMP_160]]
  // CHECK: %[[TMP_162:.*]] = mhlo.constant dense<8.5860620562778452E-15>
  // CHECK: %[[TMP_163:.*]] = mhlo.add %[[TMP_156]], %[[TMP_162]]
  // CHECK: %[[TMP_164:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_163]]
  // CHECK: %[[TMP_165:.*]] = mhlo.multiply %[[TMP_161]], %[[TMP_164]]
  // CHECK: %[[TMP_166:.*]] = mhlo.constant dense<1.400000e+01>
  // CHECK: %[[TMP_167:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_166]]
  // CHECK: %[[TMP_168:.*]] = mhlo.constant dense<1.300000e+01>
  // CHECK: %[[TMP_169:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_168]]
  // CHECK: %[[TMP_170:.*]] = mhlo.multiply %[[TMP_167]], %[[TMP_169]]
  // CHECK: %[[TMP_171:.*]] = mhlo.constant dense<-3.3896802963225832E-13>
  // CHECK: %[[TMP_172:.*]] = mhlo.add %[[TMP_165]], %[[TMP_171]]
  // CHECK: %[[TMP_173:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_172]]
  // CHECK: %[[TMP_174:.*]] = mhlo.multiply %[[TMP_170]], %[[TMP_173]]
  // CHECK: %[[TMP_175:.*]] = mhlo.constant dense<1.200000e+01>
  // CHECK: %[[TMP_176:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_175]]
  // CHECK: %[[TMP_177:.*]] = mhlo.constant dense<1.100000e+01>
  // CHECK: %[[TMP_178:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_177]]
  // CHECK: %[[TMP_179:.*]] = mhlo.multiply %[[TMP_176]], %[[TMP_178]]
  // CHECK: %[[TMP_180:.*]] = mhlo.constant dense<1.3382536530684679E-11>
  // CHECK: %[[TMP_181:.*]] = mhlo.add %[[TMP_174]], %[[TMP_180]]
  // CHECK: %[[TMP_182:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_181]]
  // CHECK: %[[TMP_183:.*]] = mhlo.multiply %[[TMP_179]], %[[TMP_182]]
  // CHECK: %[[TMP_184:.*]] = mhlo.constant dense<1.000000e+01>
  // CHECK: %[[TMP_185:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_184]]
  // CHECK: %[[TMP_186:.*]] = mhlo.constant dense<9.000000e+00>
  // CHECK: %[[TMP_187:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_186]]
  // CHECK: %[[TMP_188:.*]] = mhlo.multiply %[[TMP_185]], %[[TMP_187]]
  // CHECK: %[[TMP_189:.*]] = mhlo.constant dense<-5.2841901386874932E-10>
  // CHECK: %[[TMP_190:.*]] = mhlo.add %[[TMP_183]], %[[TMP_189]]
  // CHECK: %[[TMP_191:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_190]]
  // CHECK: %[[TMP_192:.*]] = mhlo.multiply %[[TMP_188]], %[[TMP_191]]
  // CHECK: %[[TMP_193:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_194:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_193]]
  // CHECK: %[[TMP_195:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_196:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_195]]
  // CHECK: %[[TMP_197:.*]] = mhlo.multiply %[[TMP_194]], %[[TMP_196]]
  // CHECK: %[[TMP_198:.*]] = mhlo.constant dense<2.08767569878681E-8>
  // CHECK: %[[TMP_199:.*]] = mhlo.add %[[TMP_192]], %[[TMP_198]]
  // CHECK: %[[TMP_200:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_199]]
  // CHECK: %[[TMP_201:.*]] = mhlo.multiply %[[TMP_197]], %[[TMP_200]]
  // CHECK: %[[TMP_202:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_203:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_202]]
  // CHECK: %[[TMP_204:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_205:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_204]]
  // CHECK: %[[TMP_206:.*]] = mhlo.multiply %[[TMP_203]], %[[TMP_205]]
  // CHECK: %[[TMP_207:.*]] = mhlo.constant dense<-8.2671957671957675E-7>
  // CHECK: %[[TMP_208:.*]] = mhlo.add %[[TMP_201]], %[[TMP_207]]
  // CHECK: %[[TMP_209:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_208]]
  // CHECK: %[[TMP_210:.*]] = mhlo.multiply %[[TMP_206]], %[[TMP_209]]
  // CHECK: %[[TMP_211:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_212:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_211]]
  // CHECK: %[[TMP_213:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_214:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_213]]
  // CHECK: %[[TMP_215:.*]] = mhlo.multiply %[[TMP_212]], %[[TMP_214]]
  // CHECK: %[[TMP_216:.*]] = mhlo.constant dense<3.3068783068783071E-5>
  // CHECK: %[[TMP_217:.*]] = mhlo.add %[[TMP_210]], %[[TMP_216]]
  // CHECK: %[[TMP_218:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_217]]
  // CHECK: %[[TMP_219:.*]] = mhlo.multiply %[[TMP_215]], %[[TMP_218]]
  // CHECK: %[[TMP_220:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_221:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_220]]
  // CHECK: %[[TMP_222:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_223:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_222]]
  // CHECK: %[[TMP_224:.*]] = mhlo.multiply %[[TMP_221]], %[[TMP_223]]
  // CHECK: %[[TMP_225:.*]] = mhlo.constant dense<-0.0013888888888888889>
  // CHECK: %[[TMP_226:.*]] = mhlo.add %[[TMP_219]], %[[TMP_225]]
  // CHECK: %[[TMP_227:.*]] = mhlo.multiply %[[TMP_129]], %[[TMP_226]]
  // CHECK: %[[TMP_228:.*]] = mhlo.multiply %[[TMP_224]], %[[TMP_227]]
  // CHECK: %[[TMP_229:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_230:.*]] = mhlo.divide %[[TMP_5]], %[[TMP_121]]
  // CHECK: %[[TMP_231:.*]] = mhlo.constant dense<0.083333333333333329>
  // CHECK: %[[TMP_232:.*]] = mhlo.add %[[TMP_231]], %[[TMP_228]]
  // CHECK: %[[TMP_233:.*]] = mhlo.multiply %[[TMP_230]], %[[TMP_232]]
  // CHECK: %[[TMP_234:.*]] = mhlo.add %[[TMP_229]], %[[TMP_233]]
  // CHECK: %[[TMP_235:.*]] = mhlo.multiply %[[TMP_122]], %[[TMP_234]]
  // CHECK: %[[TMP_236:.*]] = mhlo.add %[[TMP_127]], %[[TMP_235]]
  // CHECK: %[[TMP_237:.*]] = "mhlo.abs"(%[[TMP_122]])
  // CHECK: %[[TMP_238:.*]] = "mhlo.abs"(%[[TMP_120]])
  // CHECK: %[[TMP_239:.*]] = mhlo.constant dense<4.940660e-324>
  // CHECK: %[[TMP_240:.*]] = mhlo.multiply %[[TMP_238]], %[[TMP_239]]
  // CHECK: %[[TMP_241:.*]] = "mhlo.compare"(%[[TMP_237]], %[[TMP_240]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_242:.*]] = "mhlo.select"(%[[TMP_241]], %[[TMP_120]], %[[TMP_236]])
  // CHECK: %[[TMP_243:.*]] = mhlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[TMP_244:.*]] = "mhlo.compare"(%[[TMP_5]], %[[TMP_123]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_245:.*]] = "mhlo.select"(%[[TMP_244]], %[[TMP_243]], %[[TMP_242]])
  // CHECK: %[[TMP_246:.*]] = "mhlo.compare"(%[[ARG1]], %[[TMP_90]]) {comparison_direction = "LE"}
  // CHECK: %[[TMP_247:.*]] = "mhlo.floor"(%[[TMP_5]])
  // CHECK: %[[TMP_248:.*]] = "mhlo.compare"(%[[TMP_5]], %[[TMP_247]]) {comparison_direction = "NE"}
  // CHECK: %[[TMP_249:.*]] = mhlo.and %[[TMP_246]], %[[TMP_248]]
  // CHECK: %[[TMP_250:.*]] = "mhlo.select"(%[[TMP_249]], %[[TMP_243]], %[[TMP_245]])
  // CHECK: %[[TMP_251:.*]] = mhlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_252:.*]] = "mhlo.floor"(%[[ARG1]])
  // CHECK: %[[TMP_253:.*]] = "mhlo.compare"(%[[ARG1]], %[[TMP_252]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_254:.*]] = mhlo.and %[[TMP_246]], %[[TMP_253]]
  // CHECK: %[[TMP_255:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_256:.*]] = "mhlo.floor"(%[[TMP_5]])
  // CHECK: %[[TMP_257:.*]] = "mhlo.compare"(%[[TMP_5]], %[[TMP_256]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_258:.*]] = mhlo.remainder %[[TMP_5]], %[[TMP_255]]
  // CHECK: %[[TMP_259:.*]] = "mhlo.compare"(%[[TMP_258]], %[[TMP_90]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_260:.*]] = mhlo.and %[[TMP_257]], %[[TMP_259]]
  // CHECK: %[[TMP_261:.*]] = "mhlo.select"(%[[TMP_260]], %[[TMP_251]], %[[TMP_243]])
  // CHECK: %[[TMP_262:.*]] = "mhlo.select"(%[[TMP_254]], %[[TMP_261]], %[[TMP_250]])
  // CHECK: %[[TMP_263:.*]] = "mhlo.compare"(%[[TMP_5]], %[[TMP_93]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_264:.*]] = "mhlo.select"(%[[TMP_263]], %[[TMP_251]], %[[TMP_262]])
  // CHECK: %[[TMP_265:.*]] = mhlo.multiply %[[TMP_4]], %[[TMP_89]]
  // CHECK: %[[TMP_266:.*]] = mhlo.multiply %[[TMP_265]], %[[TMP_264]]
  // CHECK: %[[TMP_267:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_268:.*]] = "mhlo.compare"(%[[ARG0]], %[[TMP_267]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_269:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_270:.*]] = "mhlo.compare"(%[[ARG1]], %[[TMP_269]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_271:.*]] = "mhlo.negate"(%[[ARG1]])
  // CHECK: %[[TMP_272:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_273:.*]] = mhlo.subtract %[[ARG1]], %[[TMP_272]]
  // CHECK: %[[TMP_274:.*]] = "mhlo.select"(%[[TMP_270]], %[[TMP_271]], %[[TMP_273]])
  // CHECK: %[[TMP_275:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_276:.*]] = mhlo.constant dense<0.99999999999980993>
  // CHECK: %[[TMP_277:.*]] = mhlo.constant dense<676.5203681218851>
  // CHECK: %[[TMP_278:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_279:.*]] = mhlo.add %[[TMP_274]], %[[TMP_278]]
  // CHECK: %[[TMP_280:.*]] = mhlo.multiply %[[TMP_279]], %[[TMP_279]]
  // CHECK: %[[TMP_281:.*]] = mhlo.divide %[[TMP_277]], %[[TMP_280]]
  // CHECK: %[[TMP_282:.*]] = mhlo.subtract %[[TMP_275]], %[[TMP_281]]
  // CHECK: %[[TMP_283:.*]] = mhlo.divide %[[TMP_277]], %[[TMP_279]]
  // CHECK: %[[TMP_284:.*]] = mhlo.add %[[TMP_276]], %[[TMP_283]]
  // CHECK: %[[TMP_285:.*]] = mhlo.constant dense<-1259.1392167224028>
  // CHECK: %[[TMP_286:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_287:.*]] = mhlo.add %[[TMP_274]], %[[TMP_286]]
  // CHECK: %[[TMP_288:.*]] = mhlo.multiply %[[TMP_287]], %[[TMP_287]]
  // CHECK: %[[TMP_289:.*]] = mhlo.divide %[[TMP_285]], %[[TMP_288]]
  // CHECK: %[[TMP_290:.*]] = mhlo.subtract %[[TMP_282]], %[[TMP_289]]
  // CHECK: %[[TMP_291:.*]] = mhlo.divide %[[TMP_285]], %[[TMP_287]]
  // CHECK: %[[TMP_292:.*]] = mhlo.add %[[TMP_284]], %[[TMP_291]]
  // CHECK: %[[TMP_293:.*]] = mhlo.constant dense<771.32342877765313>
  // CHECK: %[[TMP_294:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_295:.*]] = mhlo.add %[[TMP_274]], %[[TMP_294]]
  // CHECK: %[[TMP_296:.*]] = mhlo.multiply %[[TMP_295]], %[[TMP_295]]
  // CHECK: %[[TMP_297:.*]] = mhlo.divide %[[TMP_293]], %[[TMP_296]]
  // CHECK: %[[TMP_298:.*]] = mhlo.subtract %[[TMP_290]], %[[TMP_297]]
  // CHECK: %[[TMP_299:.*]] = mhlo.divide %[[TMP_293]], %[[TMP_295]]
  // CHECK: %[[TMP_300:.*]] = mhlo.add %[[TMP_292]], %[[TMP_299]]
  // CHECK: %[[TMP_301:.*]] = mhlo.constant dense<-176.61502916214059>
  // CHECK: %[[TMP_302:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_303:.*]] = mhlo.add %[[TMP_274]], %[[TMP_302]]
  // CHECK: %[[TMP_304:.*]] = mhlo.multiply %[[TMP_303]], %[[TMP_303]]
  // CHECK: %[[TMP_305:.*]] = mhlo.divide %[[TMP_301]], %[[TMP_304]]
  // CHECK: %[[TMP_306:.*]] = mhlo.subtract %[[TMP_298]], %[[TMP_305]]
  // CHECK: %[[TMP_307:.*]] = mhlo.divide %[[TMP_301]], %[[TMP_303]]
  // CHECK: %[[TMP_308:.*]] = mhlo.add %[[TMP_300]], %[[TMP_307]]
  // CHECK: %[[TMP_309:.*]] = mhlo.constant dense<12.507343278686905>
  // CHECK: %[[TMP_310:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_311:.*]] = mhlo.add %[[TMP_274]], %[[TMP_310]]
  // CHECK: %[[TMP_312:.*]] = mhlo.multiply %[[TMP_311]], %[[TMP_311]]
  // CHECK: %[[TMP_313:.*]] = mhlo.divide %[[TMP_309]], %[[TMP_312]]
  // CHECK: %[[TMP_314:.*]] = mhlo.subtract %[[TMP_306]], %[[TMP_313]]
  // CHECK: %[[TMP_315:.*]] = mhlo.divide %[[TMP_309]], %[[TMP_311]]
  // CHECK: %[[TMP_316:.*]] = mhlo.add %[[TMP_308]], %[[TMP_315]]
  // CHECK: %[[TMP_317:.*]] = mhlo.constant dense<-0.13857109526572012>
  // CHECK: %[[TMP_318:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_319:.*]] = mhlo.add %[[TMP_274]], %[[TMP_318]]
  // CHECK: %[[TMP_320:.*]] = mhlo.multiply %[[TMP_319]], %[[TMP_319]]
  // CHECK: %[[TMP_321:.*]] = mhlo.divide %[[TMP_317]], %[[TMP_320]]
  // CHECK: %[[TMP_322:.*]] = mhlo.subtract %[[TMP_314]], %[[TMP_321]]
  // CHECK: %[[TMP_323:.*]] = mhlo.divide %[[TMP_317]], %[[TMP_319]]
  // CHECK: %[[TMP_324:.*]] = mhlo.add %[[TMP_316]], %[[TMP_323]]
  // CHECK: %[[TMP_325:.*]] = mhlo.constant dense<9.9843695780195716E-6>
  // CHECK: %[[TMP_326:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_327:.*]] = mhlo.add %[[TMP_274]], %[[TMP_326]]
  // CHECK: %[[TMP_328:.*]] = mhlo.multiply %[[TMP_327]], %[[TMP_327]]
  // CHECK: %[[TMP_329:.*]] = mhlo.divide %[[TMP_325]], %[[TMP_328]]
  // CHECK: %[[TMP_330:.*]] = mhlo.subtract %[[TMP_322]], %[[TMP_329]]
  // CHECK: %[[TMP_331:.*]] = mhlo.divide %[[TMP_325]], %[[TMP_327]]
  // CHECK: %[[TMP_332:.*]] = mhlo.add %[[TMP_324]], %[[TMP_331]]
  // CHECK: %[[TMP_333:.*]] = mhlo.constant dense<1.5056327351493116E-7>
  // CHECK: %[[TMP_334:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_335:.*]] = mhlo.add %[[TMP_274]], %[[TMP_334]]
  // CHECK: %[[TMP_336:.*]] = mhlo.multiply %[[TMP_335]], %[[TMP_335]]
  // CHECK: %[[TMP_337:.*]] = mhlo.divide %[[TMP_333]], %[[TMP_336]]
  // CHECK: %[[TMP_338:.*]] = mhlo.subtract %[[TMP_330]], %[[TMP_337]]
  // CHECK: %[[TMP_339:.*]] = mhlo.divide %[[TMP_333]], %[[TMP_335]]
  // CHECK: %[[TMP_340:.*]] = mhlo.add %[[TMP_332]], %[[TMP_339]]
  // CHECK: %[[TMP_341:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_342:.*]] = mhlo.add %[[TMP_341]], %[[TMP_274]]
  // CHECK: %[[TMP_343:.*]] = mhlo.constant dense<2.0149030205422647>
  // CHECK: %[[TMP_344:.*]] = mhlo.divide %[[TMP_274]], %[[TMP_341]]
  // CHECK: %[[TMP_345:.*]] = "mhlo.log_plus_one"(%[[TMP_344]])
  // CHECK: %[[TMP_346:.*]] = mhlo.add %[[TMP_343]], %[[TMP_345]]
  // CHECK: %[[TMP_347:.*]] = mhlo.divide %[[TMP_338]], %[[TMP_340]]
  // CHECK: %[[TMP_348:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_349:.*]] = mhlo.divide %[[TMP_348]], %[[TMP_342]]
  // CHECK: %[[TMP_350:.*]] = mhlo.add %[[TMP_346]], %[[TMP_347]]
  // CHECK: %[[TMP_351:.*]] = mhlo.subtract %[[TMP_350]], %[[TMP_349]]
  // CHECK: %[[TMP_352:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_353:.*]] = mhlo.add %[[ARG1]], %[[TMP_352]]
  // CHECK: %[[TMP_354:.*]] = "mhlo.floor"(%[[TMP_353]])
  // CHECK: %[[TMP_355:.*]] = "mhlo.abs"(%[[TMP_354]])
  // CHECK: %[[TMP_356:.*]] = mhlo.add %[[ARG1]], %[[TMP_355]]
  // CHECK: %[[TMP_357:.*]] = mhlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_358:.*]] = mhlo.multiply %[[TMP_357]], %[[TMP_356]]
  // CHECK: %[[TMP_359:.*]] = "mhlo.cosine"(%[[TMP_358]])
  // CHECK: %[[TMP_360:.*]] = "mhlo.sine"(%[[TMP_358]])
  // CHECK: %[[TMP_361:.*]] = mhlo.multiply %[[TMP_357]], %[[TMP_359]]
  // CHECK: %[[TMP_362:.*]] = mhlo.divide %[[TMP_361]], %[[TMP_360]]
  // CHECK: %[[TMP_363:.*]] = mhlo.subtract %[[TMP_351]], %[[TMP_362]]
  // CHECK: %[[TMP_364:.*]] = "mhlo.select"(%[[TMP_270]], %[[TMP_363]], %[[TMP_351]])
  // CHECK: %[[TMP_365:.*]] = "mhlo.compare"(%[[ARG1]], %[[TMP_275]]) {comparison_direction = "LE"}
  // CHECK: %[[TMP_366:.*]] = "mhlo.floor"(%[[ARG1]])
  // CHECK: %[[TMP_367:.*]] = "mhlo.compare"(%[[ARG1]], %[[TMP_366]]) {comparison_direction = "EQ"}
  // CHECK: %[[TMP_368:.*]] = mhlo.and %[[TMP_365]], %[[TMP_367]]
  // CHECK: %[[TMP_369:.*]] = mhlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[TMP_370:.*]] = "mhlo.select"(%[[TMP_368]], %[[TMP_369]], %[[TMP_364]])
  // CHECK: %[[TMP_371:.*]] = "mhlo.select"(%[[TMP_268]], %[[TMP_370]], %[[TMP_266]])
  // CHECK: %[[TMP_372:.*]] = "mhlo.floor"(%[[ARG0]])
  // CHECK: %[[TMP_373:.*]] = "mhlo.compare"(%[[ARG0]], %[[TMP_372]]) {comparison_direction = "NE"}
  // CHECK: %[[TMP_374:.*]] = "mhlo.compare"(%[[ARG0]], %[[TMP_267]]) {comparison_direction = "LT"}
  // CHECK: %[[TMP_375:.*]] = mhlo.or %[[TMP_373]], %[[TMP_374]]
  // CHECK: %[[TMP_376:.*]] = mhlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[TMP_377:.*]] = "mhlo.select"(%[[TMP_375]], %[[TMP_376]], %[[TMP_371]])
  %1 = chlo.polygamma %lhs, %rhs : tensor<f64>, tensor<f64> -> tensor<f64>
  return %1 : tensor<f64>
}

// ----

// CHECK-LABEL: @polygamma_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>, %[[ARG1:.*]]: tensor<f16>)
func @polygamma_f16(%lhs : tensor<f16>, %rhs : tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.convert"(%[[ARG0]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: "mhlo.convert"(%[[ARG1]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = "mhlo.convert"(%{{.*}}) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.polygamma %lhs, %rhs : tensor<f16>, tensor<f16> -> tensor<f16>
  return %1 : tensor<f16>
}

// ----

// CHECK-LABEL: @sinh_f32
// CHECK-SAME: (%[[X:.*]]: tensor<f32>)
func @sinh_f32(%x : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TWO:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[X]] : tensor<f32> -> tensor<?xindex>
  // CHECK: %[[CASTED_SHAPE:.*]] = tensor.cast %[[SHAPE]] : tensor<?xindex> to tensor<0xindex>
  // CHECK: %[[BROADCASTED_TWO:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TWO]], %[[CASTED_SHAPE]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<0xindex>) -> tensor<f32>
  // CHECK: %[[LOG_TWO:.*]] = "mhlo.log"(%[[BROADCASTED_TWO]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[LOG_HALF:.*]] = "mhlo.negate"(%[[LOG_TWO]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[X_PLUS_LOG_HALF:.*]] = mhlo.add %[[X]], %[[LOG_HALF]] : tensor<f32>
  // CHECK: %[[EXP_1:.*]] = "mhlo.exponential"(%[[X_PLUS_LOG_HALF]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[LOG_HALF_MINUS_X:.*]] = mhlo.subtract %[[LOG_HALF]], %[[X]] : tensor<f32>
  // CHECK: %[[EXP_2:.*]] = "mhlo.exponential"(%[[LOG_HALF_MINUS_X]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[LARGE_SINH_RESULT:.*]] = mhlo.subtract %[[EXP_1]], %[[EXP_2]] : tensor<f32>
  // CHECK: %[[EXP_X:.*]] = "mhlo.exponential"(%[[X]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[NEG_X:.*]] = "mhlo.negate"(%[[X]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[EXP_NEG_X:.*]] = "mhlo.exponential"(%[[NEG_X]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[EXP_X_MINUS_EXP_NEG_X:.*]] = mhlo.subtract %[[EXP_X]], %[[EXP_NEG_X]] : tensor<f32>
  // CHECK: %[[TWO:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK: %[[SMALL_SINH_RESULT:.*]] = mhlo.divide %[[EXP_X_MINUS_EXP_NEG_X]], %[[TWO]] : tensor<f32>
  // CHECK: %[[ABS_X:.*]] = "mhlo.abs"(%[[X]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[ABS_X_LT_ONE:.*]] = "mhlo.compare"(%[[ABS_X]], %[[ONE]]) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: %[[RESULT:.*]] = "mhlo.select"(%[[ABS_X_LT_ONE]], %[[SMALL_SINH_RESULT]], %[[LARGE_SINH_RESULT]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %[[RESULT]] : tensor<f32>
  %1 = chlo.sinh %x : tensor<f32> -> tensor<f32>
  return %1 : tensor<f32>
}

// ----

// CHECK-LABEL: @sinh_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>)
func @sinh_f16(%x : tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.convert"(%[[ARG0]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = "mhlo.convert"(%{{.*}}) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.sinh %x : tensor<f16> -> tensor<f16>
  return %1 : tensor<f16>
}

// ----

// CHECK-LABEL: @sinh_complex
// CHECK-SAME: (%[[X:.*]]: tensor<2xcomplex<f32>>)
func @sinh_complex(%x : tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK: %[[TWO:.*]] = mhlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[X]] : tensor<2xcomplex<f32>> -> tensor<?xindex>
  // CHECK: %[[CASTED_SHAPE:.*]] = tensor.cast %[[SHAPE]] : tensor<?xindex> to tensor<1xindex>
  // CHECK: %[[BROADCASTED_TWO:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TWO]], %[[CASTED_SHAPE]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<complex<f32>>, tensor<1xindex>) -> tensor<2xcomplex<f32>>
  // CHECK: %[[LOG_TWO:.*]] = "mhlo.log"(%[[BROADCASTED_TWO]]) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  // CHECK: %[[LOG_HALF:.*]] = "mhlo.negate"(%[[LOG_TWO]]) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  // CHECK: %[[X_PLUS_LOG_HALF:.*]] = mhlo.add %[[X]], %[[LOG_HALF]] : tensor<2xcomplex<f32>>
  // CHECK: %[[EXP_1:.*]] = "mhlo.exponential"(%[[X_PLUS_LOG_HALF]]) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  // CHECK: %[[LOG_HALF_MINUS_X:.*]] = mhlo.subtract %[[LOG_HALF]], %[[X]] : tensor<2xcomplex<f32>>
  // CHECK: %[[EXP_2:.*]] = "mhlo.exponential"(%[[LOG_HALF_MINUS_X]]) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  // CHECK: %[[RESULT:.*]] = mhlo.subtract %[[EXP_1]], %[[EXP_2]] : tensor<2xcomplex<f32>>
  // CHECK: return %[[RESULT]] : tensor<2xcomplex<f32>>
  %1 = chlo.sinh %x : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
  return %1 : tensor<2xcomplex<f32>>
}

// ----

// CHECK-LABEL: @cosh_f32
// CHECK-SAME: (%[[X:.*]]: tensor<f32>)
func @cosh_f32(%x : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[HALF:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[LOG_HALF:.*]] = "mhlo.log"(%[[HALF]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[X_PLUS_LOG_HALF:.*]] = mhlo.add %[[X]], %[[LOG_HALF]] : tensor<f32>
  // CHECK: %[[EXP_1:.*]] = "mhlo.exponential"(%[[X_PLUS_LOG_HALF]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[LOG_HALF_MINUS_X:.*]] = mhlo.subtract %[[LOG_HALF]], %[[X]] : tensor<f32>
  // CHECK: %[[EXP_2:.*]] = "mhlo.exponential"(%[[LOG_HALF_MINUS_X]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = mhlo.add %[[EXP_1]], %[[EXP_2]] : tensor<f32>
  // CHECK: return %[[RESULT]] : tensor<f32>
  %1 = chlo.cosh %x : tensor<f32> -> tensor<f32>
  return %1 : tensor<f32>
}

// ----

// CHECK-LABEL: @cosh_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>)
func @cosh_f16(%x : tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.convert"(%[[ARG0]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = "mhlo.convert"(%{{.*}}) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.cosh %x : tensor<f16> -> tensor<f16>
  return %1 : tensor<f16>
}

// ----

// CHECK-LABEL: @next_after_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xf32>, %[[ARG1:.*]]: tensor<2xf32>)
func @next_after_f32(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: %[[X_AS_INT:.*]] = "mhlo.bitcast_convert"(%[[ARG0]]) : (tensor<2xf32>) -> tensor<2xi32>
  // CHECK: %[[Y_AS_INT:.*]] = "mhlo.bitcast_convert"(%[[ARG1]]) : (tensor<2xf32>) -> tensor<2xi32>
  // CHECK: %[[X_IS_NAN:.*]] = "mhlo.compare"(%[[ARG0]], %[[ARG0]]) {comparison_direction = "NE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHECK: %[[Y_IS_NAN:.*]] = "mhlo.compare"(%[[ARG1]], %[[ARG1]]) {comparison_direction = "NE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHECK: %[[INPUT_IS_NAN:.*]] = mhlo.or %[[X_IS_NAN]], %[[Y_IS_NAN]] : tensor<2xi1>
  // CHECK: %[[NAN:.*]] = mhlo.constant dense<0x7FC00000> : tensor<2xf32>
  // CHECK: %[[NAN_AS_INT:.*]] = "mhlo.bitcast_convert"(%[[NAN]]) : (tensor<2xf32>) -> tensor<2xi32>
  // CHECK: %[[SIGN_MASK:.*]] = mhlo.constant dense<-2147483648> : tensor<2xi32>
  // CHECK: %[[NEGATED_SIGN_MASK:.*]] = mhlo.constant dense<2147483647> : tensor<2xi32>
  // CHECK: %[[X_ABS:.*]] = mhlo.and %[[X_AS_INT]], %[[NEGATED_SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[Y_ABS:.*]] = mhlo.and %[[Y_AS_INT]], %[[NEGATED_SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[X_AND_Y_ARE_EQUAL:.*]] = "mhlo.compare"(%[[ARG0]], %[[ARG1]]) {comparison_direction = "EQ"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHECK: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<2xi32>
  // CHECK: %[[X_ABS_IS_ZERO:.*]] = "mhlo.compare"(%[[X_ABS]], %[[ZERO]]) {comparison_direction = "EQ"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[Y_ABS_IS_ZERO:.*]] = "mhlo.compare"(%[[Y_ABS]], %[[ZERO]]) {comparison_direction = "EQ"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[X_SIGN:.*]] = mhlo.and %[[X_AS_INT]], %[[SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[Y_SIGN:.*]] = mhlo.and %[[Y_AS_INT]], %[[SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[ONE:.*]] = mhlo.constant dense<1> : tensor<2xi32>
  // CHECK: %[[RESULT_FOR_X_ZERO_Y_NON_ZERO:.*]] = mhlo.or %[[Y_SIGN]], %[[ONE]] : tensor<2xi32>
  // CHECK: %[[SIGNS_DISAGREE:.*]] = "mhlo.compare"(%[[X_SIGN]], %[[Y_SIGN]]) {comparison_direction = "NE"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[X_MAGNITUDE_LARGER_THAN_Y:.*]] = "mhlo.compare"(%[[X_ABS]], %[[Y_ABS]]) {comparison_direction = "GT"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[RESULT_HAS_SMALLER_MAGNITUDE:.*]] = mhlo.or %[[X_MAGNITUDE_LARGER_THAN_Y]], %[[SIGNS_DISAGREE]] : tensor<2xi1>
  // CHECK: %[[MINUS_ONE:.*]] = mhlo.constant dense<-1> : tensor<2xi32>
  // CHECK: %[[MAGNITUDE_ADJUSTMENT:.*]] = "mhlo.select"(%[[RESULT_HAS_SMALLER_MAGNITUDE]], %[[MINUS_ONE]], %[[ONE]]) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  // CHECK: %[[RESULT0:.*]] = mhlo.add %[[X_AS_INT]], %[[MAGNITUDE_ADJUSTMENT]] : tensor<2xi32>
  // CHECK: %[[RESULT1:.*]] = "mhlo.select"(%[[Y_ABS_IS_ZERO]], %[[Y_AS_INT]], %[[RESULT_FOR_X_ZERO_Y_NON_ZERO]]) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  // CHECK: %[[RESULT2:.*]] = "mhlo.select"(%[[X_ABS_IS_ZERO]], %[[RESULT1]], %[[RESULT0]]) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  // CHECK: %[[RESULT3:.*]] = "mhlo.select"(%[[X_AND_Y_ARE_EQUAL]], %[[Y_AS_INT]], %[[RESULT2]]) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  // CHECK: %[[RESULT4:.*]] = "mhlo.select"(%[[INPUT_IS_NAN]], %[[NAN_AS_INT]], %[[RESULT3]]) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  // CHECK: %[[FINAL_RESULT:.*]] = "mhlo.bitcast_convert"(%[[RESULT4]]) : (tensor<2xi32>) -> tensor<2xf32>
  // CHECK: return %[[FINAL_RESULT]]
  %1 = chlo.broadcast_next_after %x, %y : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}
