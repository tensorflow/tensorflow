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

// CHECK-LABEL: @erfc_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func @erfc_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.convert"(%[[ARG]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = "mhlo.convert"(%{{.*}}) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<f16>) -> tensor<f16>
  return %1 : tensor<f16>
}

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

// CHECK-LABEL: @is_pos_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func @is_pos_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  // CHECK: %[[POS_INF:.*]] = mhlo.constant dense<0x7F800000> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = "mhlo.compare"(%[[ARG]], %[[POS_INF]]) {comparison_direction = "EQ"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: return %[[RESULT]] : tensor<i1>
  %1 = chlo.is_pos_inf %arg : tensor<f32> -> tensor<i1>
  return %1 : tensor<i1>
}

// CHECK-LABEL: @is_neg_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func @is_neg_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  // CHECK: %[[NEG_INF:.*]] = mhlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = "mhlo.compare"(%[[ARG]], %[[NEG_INF]]) {comparison_direction = "EQ"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: return %[[RESULT]] : tensor<i1>
  %1 = chlo.is_neg_inf %arg : tensor<f32> -> tensor<i1>
  return %1 : tensor<i1>
}

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

// CHECK-LABEL: @lgamma_f16
// CHECK-SAME: (%[[ARG:.*]]: tensor<f16>)
func @lgamma_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.convert"(%[[ARG]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = "mhlo.convert"(%{{.*}}) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.lgamma %arg : tensor<f16> -> tensor<f16>
  return %1 : tensor<f16>
}

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

// CHECK-LABEL: @digamma_f16
// CHECK-SAME: (%[[ARG:.*]]: tensor<f16>)
func @digamma_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.convert"(%[[ARG]]) : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = "mhlo.convert"(%{{.*}}) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.digamma %arg : tensor<f16> -> tensor<f16>
  return %1 : tensor<f16>
}

// CHECK-LABEL:   func @zeta_f16(
// CHECK-SAME:                   %[[VAL_0:.*]]: tensor<f16>,
// CHECK-SAME:                   %[[VAL_1:.*]]: tensor<f16>) -> tensor<f16> {
func @zeta_f16(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
  %0 = chlo.zeta %arg0, %arg1 : (tensor<f16>, tensor<f16>) -> tensor<f16>
// CHECK: %[[VAL_2:.*]] = "mhlo.convert"(%[[VAL_0]]) : (tensor<f16>) -> tensor<f32>
// CHECK: %[[VAL_3:.*]] = "mhlo.convert"(%[[VAL_1]]) : (tensor<f16>) -> tensor<f32>
// CHECK: %[[VAL_4:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[VAL_5:.*]] = "mhlo.negate"(%[[VAL_2]]) : (tensor<f32>) -> tensor<f32>
// CHECK: %[[VAL_6:.*]] = mhlo.power %[[VAL_3]], %[[VAL_5]] : tensor<f32>
// CHECK: %[[VAL_7:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %[[VAL_8:.*]] = mhlo.add %[[VAL_3]], %[[VAL_7]] : tensor<f32>
// CHECK: %[[VAL_9:.*]] = mhlo.power %[[VAL_8]], %[[VAL_5]] : tensor<f32>
// CHECK: %[[VAL_10:.*]] = mhlo.add %[[VAL_6]], %[[VAL_9]] : tensor<f32>
// CHECK: %[[VAL_11:.*]] = mhlo.add %[[VAL_8]], %[[VAL_7]] : tensor<f32>
// CHECK: %[[VAL_12:.*]] = mhlo.power %[[VAL_11]], %[[VAL_5]] : tensor<f32>
// CHECK: %[[VAL_13:.*]] = mhlo.add %[[VAL_10]], %[[VAL_12]] : tensor<f32>
// CHECK: %[[VAL_14:.*]] = mhlo.add %[[VAL_11]], %[[VAL_7]] : tensor<f32>
// CHECK: %[[VAL_15:.*]] = mhlo.power %[[VAL_14]], %[[VAL_5]] : tensor<f32>
// CHECK: %[[VAL_16:.*]] = mhlo.add %[[VAL_13]], %[[VAL_15]] : tensor<f32>
// CHECK: %[[VAL_17:.*]] = mhlo.add %[[VAL_14]], %[[VAL_7]] : tensor<f32>
// CHECK: %[[VAL_18:.*]] = mhlo.power %[[VAL_17]], %[[VAL_5]] : tensor<f32>
// CHECK: %[[VAL_19:.*]] = mhlo.add %[[VAL_16]], %[[VAL_18]] : tensor<f32>
// CHECK: %[[VAL_20:.*]] = mhlo.add %[[VAL_17]], %[[VAL_7]] : tensor<f32>
// CHECK: %[[VAL_21:.*]] = mhlo.power %[[VAL_20]], %[[VAL_5]] : tensor<f32>
// CHECK: %[[VAL_22:.*]] = mhlo.add %[[VAL_19]], %[[VAL_21]] : tensor<f32>
// CHECK: %[[VAL_23:.*]] = mhlo.add %[[VAL_20]], %[[VAL_7]] : tensor<f32>
// CHECK: %[[VAL_24:.*]] = mhlo.power %[[VAL_23]], %[[VAL_5]] : tensor<f32>
// CHECK: %[[VAL_25:.*]] = mhlo.add %[[VAL_22]], %[[VAL_24]] : tensor<f32>
// CHECK: %[[VAL_26:.*]] = mhlo.add %[[VAL_23]], %[[VAL_7]] : tensor<f32>
// CHECK: %[[VAL_27:.*]] = mhlo.power %[[VAL_26]], %[[VAL_5]] : tensor<f32>
// CHECK: %[[VAL_28:.*]] = mhlo.add %[[VAL_25]], %[[VAL_27]] : tensor<f32>
// CHECK: %[[VAL_29:.*]] = mhlo.add %[[VAL_26]], %[[VAL_7]] : tensor<f32>
// CHECK: %[[VAL_30:.*]] = mhlo.power %[[VAL_29]], %[[VAL_5]] : tensor<f32>
// CHECK: %[[VAL_31:.*]] = mhlo.add %[[VAL_28]], %[[VAL_30]] : tensor<f32>
// CHECK: %[[VAL_32:.*]] = mhlo.add %[[VAL_29]], %[[VAL_7]] : tensor<f32>
// CHECK: %[[VAL_33:.*]] = mhlo.power %[[VAL_32]], %[[VAL_5]] : tensor<f32>
// CHECK: %[[VAL_34:.*]] = mhlo.add %[[VAL_31]], %[[VAL_33]] : tensor<f32>
// CHECK: %[[VAL_35:.*]] = mhlo.add %[[VAL_32]], %[[VAL_7]] : tensor<f32>
// CHECK: %[[VAL_36:.*]] = mhlo.power %[[VAL_35]], %[[VAL_5]] : tensor<f32>
// CHECK: %[[VAL_37:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %[[VAL_38:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_37]] : tensor<f32>
// CHECK: %[[VAL_39:.*]] = mhlo.multiply %[[VAL_36]], %[[VAL_35]] : tensor<f32>
// CHECK: %[[VAL_40:.*]] = mhlo.divide %[[VAL_39]], %[[VAL_38]] : tensor<f32>
// CHECK: %[[VAL_41:.*]] = mhlo.add %[[VAL_34]], %[[VAL_40]] : tensor<f32>
// CHECK: %[[VAL_42:.*]] = mhlo.multiply %[[VAL_35]], %[[VAL_35]] : tensor<f32>
// CHECK: %[[VAL_43:.*]] = mhlo.divide %[[VAL_7]], %[[VAL_42]] : tensor<f32>
// CHECK: %[[VAL_44:.*]] = mhlo.constant dense<2.200000e+01> : tensor<f32>
// CHECK: %[[VAL_45:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_44]] : tensor<f32>
// CHECK: %[[VAL_46:.*]] = mhlo.constant dense<2.100000e+01> : tensor<f32>
// CHECK: %[[VAL_47:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_46]] : tensor<f32>
// CHECK: %[[VAL_48:.*]] = mhlo.multiply %[[VAL_45]], %[[VAL_47]] : tensor<f32>
// CHECK: %[[VAL_49:.*]] = mhlo.constant dense<-1.39544646E-19> : tensor<f32>
// CHECK: %[[VAL_50:.*]] = mhlo.add %[[VAL_4]], %[[VAL_49]] : tensor<f32>
// CHECK: %[[VAL_51:.*]] = mhlo.multiply %[[VAL_43]], %[[VAL_50]] : tensor<f32>
// CHECK: %[[VAL_52:.*]] = mhlo.multiply %[[VAL_48]], %[[VAL_51]] : tensor<f32>
// CHECK: %[[VAL_53:.*]] = mhlo.constant dense<2.000000e+01> : tensor<f32>
// CHECK: %[[VAL_54:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_53]] : tensor<f32>
// CHECK: %[[VAL_55:.*]] = mhlo.constant dense<1.900000e+01> : tensor<f32>
// CHECK: %[[VAL_56:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_55]] : tensor<f32>
// CHECK: %[[VAL_57:.*]] = mhlo.multiply %[[VAL_54]], %[[VAL_56]] : tensor<f32>
// CHECK: %[[VAL_58:.*]] = mhlo.constant dense<5.50900303E-18> : tensor<f32>
// CHECK: %[[VAL_59:.*]] = mhlo.add %[[VAL_52]], %[[VAL_58]] : tensor<f32>
// CHECK: %[[VAL_60:.*]] = mhlo.multiply %[[VAL_43]], %[[VAL_59]] : tensor<f32>
// CHECK: %[[VAL_61:.*]] = mhlo.multiply %[[VAL_57]], %[[VAL_60]] : tensor<f32>
// CHECK: %[[VAL_62:.*]] = mhlo.constant dense<1.800000e+01> : tensor<f32>
// CHECK: %[[VAL_63:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_62]] : tensor<f32>
// CHECK: %[[VAL_64:.*]] = mhlo.constant dense<1.700000e+01> : tensor<f32>
// CHECK: %[[VAL_65:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_64]] : tensor<f32>
// CHECK: %[[VAL_66:.*]] = mhlo.multiply %[[VAL_63]], %[[VAL_65]] : tensor<f32>
// CHECK: %[[VAL_67:.*]] = mhlo.constant dense<-2.17486866E-16> : tensor<f32>
// CHECK: %[[VAL_68:.*]] = mhlo.add %[[VAL_61]], %[[VAL_67]] : tensor<f32>
// CHECK: %[[VAL_69:.*]] = mhlo.multiply %[[VAL_43]], %[[VAL_68]] : tensor<f32>
// CHECK: %[[VAL_70:.*]] = mhlo.multiply %[[VAL_66]], %[[VAL_69]] : tensor<f32>
// CHECK: %[[VAL_71:.*]] = mhlo.constant dense<1.600000e+01> : tensor<f32>
// CHECK: %[[VAL_72:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_71]] : tensor<f32>
// CHECK: %[[VAL_73:.*]] = mhlo.constant dense<1.500000e+01> : tensor<f32>
// CHECK: %[[VAL_74:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_73]] : tensor<f32>
// CHECK: %[[VAL_75:.*]] = mhlo.multiply %[[VAL_72]], %[[VAL_74]] : tensor<f32>
// CHECK: %[[VAL_76:.*]] = mhlo.constant dense<8.58606213E-15> : tensor<f32>
// CHECK: %[[VAL_77:.*]] = mhlo.add %[[VAL_70]], %[[VAL_76]] : tensor<f32>
// CHECK: %[[VAL_78:.*]] = mhlo.multiply %[[VAL_43]], %[[VAL_77]] : tensor<f32>
// CHECK: %[[VAL_79:.*]] = mhlo.multiply %[[VAL_75]], %[[VAL_78]] : tensor<f32>
// CHECK: %[[VAL_80:.*]] = mhlo.constant dense<1.400000e+01> : tensor<f32>
// CHECK: %[[VAL_81:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_80]] : tensor<f32>
// CHECK: %[[VAL_82:.*]] = mhlo.constant dense<1.300000e+01> : tensor<f32>
// CHECK: %[[VAL_83:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_82]] : tensor<f32>
// CHECK: %[[VAL_84:.*]] = mhlo.multiply %[[VAL_81]], %[[VAL_83]] : tensor<f32>
// CHECK: %[[VAL_85:.*]] = mhlo.constant dense<-3.3896803E-13> : tensor<f32>
// CHECK: %[[VAL_86:.*]] = mhlo.add %[[VAL_79]], %[[VAL_85]] : tensor<f32>
// CHECK: %[[VAL_87:.*]] = mhlo.multiply %[[VAL_43]], %[[VAL_86]] : tensor<f32>
// CHECK: %[[VAL_88:.*]] = mhlo.multiply %[[VAL_84]], %[[VAL_87]] : tensor<f32>
// CHECK: %[[VAL_89:.*]] = mhlo.constant dense<1.200000e+01> : tensor<f32>
// CHECK: %[[VAL_90:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_89]] : tensor<f32>
// CHECK: %[[VAL_91:.*]] = mhlo.constant dense<1.100000e+01> : tensor<f32>
// CHECK: %[[VAL_92:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_91]] : tensor<f32>
// CHECK: %[[VAL_93:.*]] = mhlo.multiply %[[VAL_90]], %[[VAL_92]] : tensor<f32>
// CHECK: %[[VAL_94:.*]] = mhlo.constant dense<1.33825364E-11> : tensor<f32>
// CHECK: %[[VAL_95:.*]] = mhlo.add %[[VAL_88]], %[[VAL_94]] : tensor<f32>
// CHECK: %[[VAL_96:.*]] = mhlo.multiply %[[VAL_43]], %[[VAL_95]] : tensor<f32>
// CHECK: %[[VAL_97:.*]] = mhlo.multiply %[[VAL_93]], %[[VAL_96]] : tensor<f32>
// CHECK: %[[VAL_98:.*]] = mhlo.constant dense<1.000000e+01> : tensor<f32>
// CHECK: %[[VAL_99:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_98]] : tensor<f32>
// CHECK: %[[VAL_100:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK: %[[VAL_101:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_100]] : tensor<f32>
// CHECK: %[[VAL_102:.*]] = mhlo.multiply %[[VAL_99]], %[[VAL_101]] : tensor<f32>
// CHECK: %[[VAL_103:.*]] = mhlo.constant dense<-5.28419031E-10> : tensor<f32>
// CHECK: %[[VAL_104:.*]] = mhlo.add %[[VAL_97]], %[[VAL_103]] : tensor<f32>
// CHECK: %[[VAL_105:.*]] = mhlo.multiply %[[VAL_43]], %[[VAL_104]] : tensor<f32>
// CHECK: %[[VAL_106:.*]] = mhlo.multiply %[[VAL_102]], %[[VAL_105]] : tensor<f32>
// CHECK: %[[VAL_107:.*]] = mhlo.constant dense<8.000000e+00> : tensor<f32>
// CHECK: %[[VAL_108:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_107]] : tensor<f32>
// CHECK: %[[VAL_109:.*]] = mhlo.constant dense<7.000000e+00> : tensor<f32>
// CHECK: %[[VAL_110:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_109]] : tensor<f32>
// CHECK: %[[VAL_111:.*]] = mhlo.multiply %[[VAL_108]], %[[VAL_110]] : tensor<f32>
// CHECK: %[[VAL_112:.*]] = mhlo.constant dense<2.08767563E-8> : tensor<f32>
// CHECK: %[[VAL_113:.*]] = mhlo.add %[[VAL_106]], %[[VAL_112]] : tensor<f32>
// CHECK: %[[VAL_114:.*]] = mhlo.multiply %[[VAL_43]], %[[VAL_113]] : tensor<f32>
// CHECK: %[[VAL_115:.*]] = mhlo.multiply %[[VAL_111]], %[[VAL_114]] : tensor<f32>
// CHECK: %[[VAL_116:.*]] = mhlo.constant dense<6.000000e+00> : tensor<f32>
// CHECK: %[[VAL_117:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_116]] : tensor<f32>
// CHECK: %[[VAL_118:.*]] = mhlo.constant dense<5.000000e+00> : tensor<f32>
// CHECK: %[[VAL_119:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_118]] : tensor<f32>
// CHECK: %[[VAL_120:.*]] = mhlo.multiply %[[VAL_117]], %[[VAL_119]] : tensor<f32>
// CHECK: %[[VAL_121:.*]] = mhlo.constant dense<-8.26719599E-7> : tensor<f32>
// CHECK: %[[VAL_122:.*]] = mhlo.add %[[VAL_115]], %[[VAL_121]] : tensor<f32>
// CHECK: %[[VAL_123:.*]] = mhlo.multiply %[[VAL_43]], %[[VAL_122]] : tensor<f32>
// CHECK: %[[VAL_124:.*]] = mhlo.multiply %[[VAL_120]], %[[VAL_123]] : tensor<f32>
// CHECK: %[[VAL_125:.*]] = mhlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK: %[[VAL_126:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_125]] : tensor<f32>
// CHECK: %[[VAL_127:.*]] = mhlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK: %[[VAL_128:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_127]] : tensor<f32>
// CHECK: %[[VAL_129:.*]] = mhlo.multiply %[[VAL_126]], %[[VAL_128]] : tensor<f32>
// CHECK: %[[VAL_130:.*]] = mhlo.constant dense<3.30687835E-5> : tensor<f32>
// CHECK: %[[VAL_131:.*]] = mhlo.add %[[VAL_124]], %[[VAL_130]] : tensor<f32>
// CHECK: %[[VAL_132:.*]] = mhlo.multiply %[[VAL_43]], %[[VAL_131]] : tensor<f32>
// CHECK: %[[VAL_133:.*]] = mhlo.multiply %[[VAL_129]], %[[VAL_132]] : tensor<f32>
// CHECK: %[[VAL_134:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK: %[[VAL_135:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_134]] : tensor<f32>
// CHECK: %[[VAL_136:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %[[VAL_137:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_136]] : tensor<f32>
// CHECK: %[[VAL_138:.*]] = mhlo.multiply %[[VAL_135]], %[[VAL_137]] : tensor<f32>
// CHECK: %[[VAL_139:.*]] = mhlo.constant dense<-0.00138888892> : tensor<f32>
// CHECK: %[[VAL_140:.*]] = mhlo.add %[[VAL_133]], %[[VAL_139]] : tensor<f32>
// CHECK: %[[VAL_141:.*]] = mhlo.multiply %[[VAL_43]], %[[VAL_140]] : tensor<f32>
// CHECK: %[[VAL_142:.*]] = mhlo.multiply %[[VAL_138]], %[[VAL_141]] : tensor<f32>
// CHECK: %[[VAL_143:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK: %[[VAL_144:.*]] = mhlo.divide %[[VAL_2]], %[[VAL_35]] : tensor<f32>
// CHECK: %[[VAL_145:.*]] = mhlo.constant dense<0.0833333358> : tensor<f32>
// CHECK: %[[VAL_146:.*]] = mhlo.add %[[VAL_145]], %[[VAL_142]] : tensor<f32>
// CHECK: %[[VAL_147:.*]] = mhlo.multiply %[[VAL_144]], %[[VAL_146]] : tensor<f32>
// CHECK: %[[VAL_148:.*]] = mhlo.add %[[VAL_143]], %[[VAL_147]] : tensor<f32>
// CHECK: %[[VAL_149:.*]] = mhlo.multiply %[[VAL_36]], %[[VAL_148]] : tensor<f32>
// CHECK: %[[VAL_150:.*]] = mhlo.add %[[VAL_41]], %[[VAL_149]] : tensor<f32>
// CHECK: %[[VAL_151:.*]] = "mhlo.abs"(%[[VAL_36]]) : (tensor<f32>) -> tensor<f32>
// CHECK: %[[VAL_152:.*]] = "mhlo.abs"(%[[VAL_34]]) : (tensor<f32>) -> tensor<f32>
// CHECK: %[[VAL_153:.*]] = mhlo.constant dense<1.401300e-45> : tensor<f32>
// CHECK: %[[VAL_154:.*]] = mhlo.multiply %[[VAL_152]], %[[VAL_153]] : tensor<f32>
// CHECK: %[[VAL_155:.*]] = "mhlo.compare"(%[[VAL_151]], %[[VAL_154]]) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[VAL_156:.*]] = "mhlo.select"(%[[VAL_155]], %[[VAL_34]], %[[VAL_150]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[VAL_157:.*]] = mhlo.constant dense<0x7F800000> : tensor<f32>
// CHECK: %[[VAL_158:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_37]]) {comparison_direction = "EQ"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[VAL_159:.*]] = "mhlo.select"(%[[VAL_158]], %[[VAL_157]], %[[VAL_156]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[VAL_160:.*]] = mhlo.constant dense<0x7FC00000> : tensor<f32>
// CHECK: %[[VAL_161:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_37]]) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[VAL_162:.*]] = "mhlo.select"(%[[VAL_161]], %[[VAL_160]], %[[VAL_159]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[VAL_163:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[VAL_164:.*]] = "mhlo.compare"(%[[VAL_3]], %[[VAL_163]]) {comparison_direction = "LE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[VAL_165:.*]] = "mhlo.floor"(%[[VAL_2]]) : (tensor<f32>) -> tensor<f32>
// CHECK: %[[VAL_166:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_165]]) {comparison_direction = "NE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[VAL_167:.*]] = mhlo.and %[[VAL_164]], %[[VAL_166]] : tensor<i1>
// CHECK: %[[VAL_169:.*]] = "mhlo.floor"(%[[VAL_3]]) : (tensor<f32>) -> tensor<f32>
// CHECK: %[[VAL_170:.*]] = "mhlo.compare"(%[[VAL_3]], %[[VAL_169]]) {comparison_direction = "EQ"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[VAL_171:.*]] = mhlo.and %[[VAL_164]], %[[VAL_170]] : tensor<i1>
// CHECK: %[[VAL_172:.*]] = "mhlo.select"(%[[VAL_171]], %[[VAL_157]], %[[VAL_162]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[VAL_173:.*]] = "mhlo.select"(%[[VAL_167]], %[[VAL_160]], %[[VAL_172]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[VAL_174:.*]] = "mhlo.convert"(%[[VAL_173]]) : (tensor<f32>) -> tensor<f16>
  return %0 : tensor<f16>
// CHECK: return %[[VAL_174]] : tensor<f16>
}
