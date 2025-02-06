// RUN: mlir-hlo-opt --chlo-legalize-to-hlo --split-input-file -verify-diagnostics %s | FileCheck %s --dump-input-context=20
// RUN: mlir-hlo-opt --chlo-legalize-to-high-level-mhlo --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=CHECK-HIGH-LEVEL

// CHECK-LABEL: func.func @asin_bf16(
// CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<bf16>
// CHECK-NEXT:     %[[TMP_1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<bf16>
// CHECK-NEXT:     %[[TMP_2:.*]] = mhlo.subtract %[[TMP_1]], %[[TMP_arg0]] : tensor<bf16>
// CHECK-NEXT:     %[[TMP_3:.*]] = mhlo.add %[[TMP_1]], %[[TMP_arg0]] : tensor<bf16>
// CHECK-NEXT:     %[[TMP_4:.*]] = mhlo.multiply %[[TMP_2]], %[[TMP_3]] : tensor<bf16>
// CHECK-NEXT:     %[[TMP_5:.*]] = mhlo.sqrt %[[TMP_4]] : tensor<bf16>
// CHECK-NEXT:     %[[TMP_6:.*]] = mhlo.add %[[TMP_1]], %[[TMP_5]] : tensor<bf16>
// CHECK-NEXT:     %[[TMP_7:.*]] = mhlo.atan2 %[[TMP_arg0]], %[[TMP_6]] : tensor<bf16>
// CHECK-NEXT:     %[[TMP_8:.*]] = mhlo.add %[[TMP_7]], %[[TMP_7]] : tensor<bf16>
// CHECK-NEXT:     return %[[TMP_8]] : tensor<bf16>
func.func @asin_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  %result = "chlo.asin"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %result : tensor<bf16>
}

// -----

// CHECK-LABEL: func.func @asin_f16(
// CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<f16>
// CHECK-NEXT:    %[[TMP_1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f16>
// CHECK-NEXT:    %[[TMP_2:.*]] = mhlo.subtract %[[TMP_1]], %[[TMP_arg0]] : tensor<f16>
// CHECK-NEXT:    %[[TMP_3:.*]] = mhlo.add %[[TMP_1]], %[[TMP_arg0]] : tensor<f16>
// CHECK-NEXT:    %[[TMP_4:.*]] = mhlo.multiply %[[TMP_2]], %[[TMP_3]] : tensor<f16>
// CHECK-NEXT:    %[[TMP_5:.*]] = mhlo.sqrt %[[TMP_4]] : tensor<f16>
// CHECK-NEXT:    %[[TMP_6:.*]] = mhlo.add %[[TMP_1]], %[[TMP_5]] : tensor<f16>
// CHECK-NEXT:    %[[TMP_7:.*]] = mhlo.atan2 %[[TMP_arg0]], %[[TMP_6]] : tensor<f16>
// CHECK-NEXT:    %[[TMP_8:.*]] = mhlo.add %[[TMP_7]], %[[TMP_7]] : tensor<f16>
// CHECK-NEXT:    return %[[TMP_8]] : tensor<f16>
func.func @asin_f16(%arg : tensor<f16>) -> tensor<f16> {
  %result = "chlo.asin"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %result : tensor<f16>
}

// -----

// CHECK-LABEL: func.func @asin_f32(
// CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %[[TMP_1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[TMP_2:.*]] = mhlo.subtract %[[TMP_1]], %[[TMP_arg0]] : tensor<f32>
// CHECK-NEXT:    %[[TMP_3:.*]] = mhlo.add %[[TMP_1]], %[[TMP_arg0]] : tensor<f32>
// CHECK-NEXT:    %[[TMP_4:.*]] = mhlo.multiply %[[TMP_2]], %[[TMP_3]] : tensor<f32>
// CHECK-NEXT:    %[[TMP_5:.*]] = mhlo.sqrt %[[TMP_4]] : tensor<f32>
// CHECK-NEXT:    %[[TMP_6:.*]] = mhlo.add %[[TMP_1]], %[[TMP_5]] : tensor<f32>
// CHECK-NEXT:    %[[TMP_7:.*]] = mhlo.atan2 %[[TMP_arg0]], %[[TMP_6]] : tensor<f32>
// CHECK-NEXT:    %[[TMP_8:.*]] = mhlo.add %[[TMP_7]], %[[TMP_7]] : tensor<f32>
// CHECK-NEXT:    return %[[TMP_8]] : tensor<f32>
func.func @asin_f32(%arg : tensor<f32>) -> tensor<f32> {
  %result = "chlo.asin"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// -----

// CHECK-LABEL:  func.func @asin_f64(
// CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[TMP_1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[TMP_2:.*]] = mhlo.subtract %[[TMP_1]], %[[TMP_arg0]] : tensor<f64>
// CHECK-NEXT:    %[[TMP_3:.*]] = mhlo.add %[[TMP_1]], %[[TMP_arg0]] : tensor<f64>
// CHECK-NEXT:    %[[TMP_4:.*]] = mhlo.multiply %[[TMP_2]], %[[TMP_3]] : tensor<f64>
// CHECK-NEXT:    %[[TMP_5:.*]] = mhlo.sqrt %[[TMP_4]] : tensor<f64>
// CHECK-NEXT:    %[[TMP_6:.*]] = mhlo.add %[[TMP_1]], %[[TMP_5]] : tensor<f64>
// CHECK-NEXT:    %[[TMP_7:.*]] = mhlo.atan2 %[[TMP_arg0]], %[[TMP_6]] : tensor<f64>
// CHECK-NEXT:    %[[TMP_8:.*]] = mhlo.add %[[TMP_7]], %[[TMP_7]] : tensor<f64>
// CHECK-NEXT:    return %[[TMP_8]] : tensor<f64>
func.func @asin_f64(%arg : tensor<f64>) -> tensor<f64> {
  %result = "chlo.asin"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %result : tensor<f64>
}

// -----

// CHECK-LABEL: func.func @asin_complex_f32(
// CHECK-SAME:   %[[TEMP_arg0:.*]]: tensor<complex<f32>>) -> tensor<complex<f32>>
// CHECK: %[[TEMP_0:.*]] = mhlo.real %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_1:.*]] = mhlo.real %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_2:.*]] = mhlo.abs %[[TEMP_1]] : tensor<f32>
// CHECK: %[[TEMP_3:.*]] = mhlo.imag %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_4:.*]] = mhlo.abs %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_5:.*]] = mhlo.maximum %[[TEMP_2]], %[[TEMP_4]] : tensor<f32>
// CHECK: %[[TEMP_6:.*]] = mhlo.constant dense<3.40282347E+38> : tensor<f32>
// CHECK: %[[TEMP_7:.*]] = mhlo.sqrt %[[TEMP_6]] : tensor<f32>
// CHECK: %[[TEMP_8:.*]] = mhlo.constant dense<8.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_9:.*]] = mhlo.divide %[[TEMP_7]], %[[TEMP_8]] : tensor<f32>
// CHECK: %[[TEMP_10:.*]] = mhlo.compare  GE, %[[TEMP_5]], %[[TEMP_9]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_11:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_12:.*]] = mhlo.compare  LE, %[[TEMP_2]], %[[TEMP_11]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_13:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK: %[[TEMP_14:.*]] = mhlo.add %[[TEMP_2]], %[[TEMP_11]] : tensor<f32>
// CHECK: %[[TEMP_15:.*]] = mhlo.abs %[[TEMP_14]] : tensor<f32>
// CHECK: %[[TEMP_16:.*]] = mhlo.maximum %[[TEMP_15]], %[[TEMP_4]] : tensor<f32>
// CHECK: %[[TEMP_17:.*]] = mhlo.minimum %[[TEMP_15]], %[[TEMP_4]] : tensor<f32>
// CHECK: %[[TEMP_18:.*]] = mhlo.compare  EQ, %[[TEMP_16]], %[[TEMP_17]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_19:.*]] = mhlo.constant dense<1.41421354> : tensor<f32>
// CHECK: %[[TEMP_20:.*]] = mhlo.multiply %[[TEMP_19]], %[[TEMP_16]] : tensor<f32>
// CHECK: %[[TEMP_21:.*]] = mhlo.divide %[[TEMP_17]], %[[TEMP_16]] : tensor<f32>
// CHECK: %[[TEMP_22:.*]] = mhlo.multiply %[[TEMP_21]], %[[TEMP_21]] : tensor<f32>
// CHECK: %[[TEMP_23:.*]] = mhlo.add %[[TEMP_11]], %[[TEMP_22]] : tensor<f32>
// CHECK: %[[TEMP_24:.*]] = mhlo.sqrt %[[TEMP_23]] : tensor<f32>
// CHECK: %[[TEMP_25:.*]] = mhlo.compare  EQ, %[[TEMP_24]], %[[TEMP_11]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_26:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_27:.*]] = mhlo.compare  GT, %[[TEMP_22]], %[[TEMP_26]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_28:.*]] = mhlo.and %[[TEMP_25]], %[[TEMP_27]] : tensor<i1>
// CHECK: %[[TEMP_29:.*]] = mhlo.multiply %[[TEMP_16]], %[[TEMP_22]] : tensor<f32>
// CHECK: %[[TEMP_30:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_31:.*]] = mhlo.divide %[[TEMP_29]], %[[TEMP_30]] : tensor<f32>
// CHECK: %[[TEMP_32:.*]] = mhlo.add %[[TEMP_16]], %[[TEMP_31]] : tensor<f32>
// CHECK: %[[TEMP_33:.*]] = mhlo.multiply %[[TEMP_16]], %[[TEMP_24]] : tensor<f32>
// CHECK: %[[TEMP_34:.*]] = mhlo.select %[[TEMP_28]], %[[TEMP_32]], %[[TEMP_33]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_35:.*]] = mhlo.select %[[TEMP_18]], %[[TEMP_20]], %[[TEMP_34]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_36:.*]] = mhlo.subtract %[[TEMP_2]], %[[TEMP_11]] : tensor<f32>
// CHECK: %[[TEMP_37:.*]] = mhlo.abs %[[TEMP_36]] : tensor<f32>
// CHECK: %[[TEMP_38:.*]] = mhlo.maximum %[[TEMP_37]], %[[TEMP_4]] : tensor<f32>
// CHECK: %[[TEMP_39:.*]] = mhlo.minimum %[[TEMP_37]], %[[TEMP_4]] : tensor<f32>
// CHECK: %[[TEMP_40:.*]] = mhlo.compare  EQ, %[[TEMP_38]], %[[TEMP_39]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_41:.*]] = mhlo.multiply %[[TEMP_19]], %[[TEMP_38]] : tensor<f32>
// CHECK: %[[TEMP_42:.*]] = mhlo.divide %[[TEMP_39]], %[[TEMP_38]] : tensor<f32>
// CHECK: %[[TEMP_43:.*]] = mhlo.multiply %[[TEMP_42]], %[[TEMP_42]] : tensor<f32>
// CHECK: %[[TEMP_44:.*]] = mhlo.add %[[TEMP_11]], %[[TEMP_43]] : tensor<f32>
// CHECK: %[[TEMP_45:.*]] = mhlo.sqrt %[[TEMP_44]] : tensor<f32>
// CHECK: %[[TEMP_46:.*]] = mhlo.compare  EQ, %[[TEMP_45]], %[[TEMP_11]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_47:.*]] = mhlo.compare  GT, %[[TEMP_43]], %[[TEMP_26]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_48:.*]] = mhlo.and %[[TEMP_46]], %[[TEMP_47]] : tensor<i1>
// CHECK: %[[TEMP_49:.*]] = mhlo.multiply %[[TEMP_38]], %[[TEMP_43]] : tensor<f32>
// CHECK: %[[TEMP_50:.*]] = mhlo.divide %[[TEMP_49]], %[[TEMP_30]] : tensor<f32>
// CHECK: %[[TEMP_51:.*]] = mhlo.add %[[TEMP_38]], %[[TEMP_50]] : tensor<f32>
// CHECK: %[[TEMP_52:.*]] = mhlo.multiply %[[TEMP_38]], %[[TEMP_45]] : tensor<f32>
// CHECK: %[[TEMP_53:.*]] = mhlo.select %[[TEMP_48]], %[[TEMP_51]], %[[TEMP_52]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_54:.*]] = mhlo.select %[[TEMP_40]], %[[TEMP_41]], %[[TEMP_53]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_55:.*]] = mhlo.add %[[TEMP_35]], %[[TEMP_54]] : tensor<f32>
// CHECK: %[[TEMP_56:.*]] = mhlo.multiply %[[TEMP_13]], %[[TEMP_55]] : tensor<f32>
// CHECK: %[[TEMP_57:.*]] = mhlo.add %[[TEMP_56]], %[[TEMP_2]] : tensor<f32>
// CHECK: %[[TEMP_58:.*]] = mhlo.multiply %[[TEMP_13]], %[[TEMP_57]] : tensor<f32>
// CHECK: %[[TEMP_59:.*]] = mhlo.multiply %[[TEMP_4]], %[[TEMP_4]] : tensor<f32>
// CHECK: %[[TEMP_60:.*]] = mhlo.add %[[TEMP_35]], %[[TEMP_14]] : tensor<f32>
// CHECK: %[[TEMP_61:.*]] = mhlo.divide %[[TEMP_59]], %[[TEMP_60]] : tensor<f32>
// CHECK: %[[TEMP_62:.*]] = mhlo.subtract %[[TEMP_54]], %[[TEMP_36]] : tensor<f32>
// CHECK: %[[TEMP_63:.*]] = mhlo.add %[[TEMP_61]], %[[TEMP_62]] : tensor<f32>
// CHECK: %[[TEMP_64:.*]] = mhlo.multiply %[[TEMP_58]], %[[TEMP_63]] : tensor<f32>
// CHECK: %[[TEMP_65:.*]] = mhlo.sqrt %[[TEMP_64]] : tensor<f32>
// CHECK: %[[TEMP_66:.*]] = mhlo.divide %[[TEMP_58]], %[[TEMP_60]] : tensor<f32>
// CHECK: %[[TEMP_67:.*]] = mhlo.add %[[TEMP_54]], %[[TEMP_36]] : tensor<f32>
// CHECK: %[[TEMP_68:.*]] = mhlo.divide %[[TEMP_58]], %[[TEMP_67]] : tensor<f32>
// CHECK: %[[TEMP_69:.*]] = mhlo.add %[[TEMP_66]], %[[TEMP_68]] : tensor<f32>
// CHECK: %[[TEMP_70:.*]] = mhlo.sqrt %[[TEMP_69]] : tensor<f32>
// CHECK: %[[TEMP_71:.*]] = mhlo.multiply %[[TEMP_4]], %[[TEMP_70]] : tensor<f32>
// CHECK: %[[TEMP_72:.*]] = mhlo.select %[[TEMP_12]], %[[TEMP_65]], %[[TEMP_71]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_73:.*]] = mhlo.select %[[TEMP_10]], %[[TEMP_4]], %[[TEMP_72]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_74:.*]] = mhlo.constant dense<9.99999995E+11> : tensor<f32>
// CHECK: %[[TEMP_75:.*]] = mhlo.multiply %[[TEMP_9]], %[[TEMP_74]] : tensor<f32>
// CHECK: %[[TEMP_76:.*]] = mhlo.compare  LT, %[[TEMP_2]], %[[TEMP_75]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_77:.*]] = mhlo.constant dense<9.99999997E-7> : tensor<f32>
// CHECK: %[[TEMP_78:.*]] = mhlo.multiply %[[TEMP_9]], %[[TEMP_77]] : tensor<f32>
// CHECK: %[[TEMP_79:.*]] = mhlo.constant dense<1.000000e+02> : tensor<f32>
// CHECK: %[[TEMP_80:.*]] = mhlo.multiply %[[TEMP_9]], %[[TEMP_79]] : tensor<f32>
// CHECK: %[[TEMP_81:.*]] = mhlo.select %[[TEMP_76]], %[[TEMP_78]], %[[TEMP_80]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_82:.*]] = mhlo.compare  GE, %[[TEMP_4]], %[[TEMP_81]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_83:.*]] = mhlo.select %[[TEMP_82]], %[[TEMP_4]], %[[TEMP_2]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_84:.*]] = mhlo.select %[[TEMP_82]], %[[TEMP_81]], %[[TEMP_9]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_85:.*]] = mhlo.compare  GE, %[[TEMP_83]], %[[TEMP_84]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_86:.*]] = mhlo.log %[[TEMP_30]] : tensor<f32>
// CHECK: %[[TEMP_87:.*]] = mhlo.log %[[TEMP_83]] : tensor<f32>
// CHECK: %[[TEMP_88:.*]] = mhlo.add %[[TEMP_86]], %[[TEMP_87]] : tensor<f32>
// CHECK: %[[TEMP_89:.*]] = mhlo.constant dense<0x7F800000> : tensor<f32>
// CHECK: %[[TEMP_90:.*]] = mhlo.compare  EQ, %[[TEMP_4]], %[[TEMP_89]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_91:.*]] = mhlo.not %[[TEMP_90]] : tensor<i1>
// CHECK: %[[TEMP_92:.*]] = mhlo.and %[[TEMP_82]], %[[TEMP_91]] : tensor<i1>
// CHECK: %[[TEMP_93:.*]] = mhlo.divide %[[TEMP_2]], %[[TEMP_4]] : tensor<f32>
// CHECK: %[[TEMP_94:.*]] = mhlo.select %[[TEMP_92]], %[[TEMP_93]], %[[TEMP_26]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_95:.*]] = mhlo.multiply %[[TEMP_94]], %[[TEMP_94]] : tensor<f32>
// CHECK: %[[TEMP_96:.*]] = mhlo.log_plus_one %[[TEMP_95]] : tensor<f32>
// CHECK: %[[TEMP_97:.*]] = mhlo.multiply %[[TEMP_13]], %[[TEMP_96]] : tensor<f32>
// CHECK: %[[TEMP_98:.*]] = mhlo.add %[[TEMP_88]], %[[TEMP_97]] : tensor<f32>
// CHECK: %[[TEMP_99:.*]] = mhlo.constant dense<1.17549435E-38> : tensor<f32>
// CHECK: %[[TEMP_100:.*]] = mhlo.sqrt %[[TEMP_99]] : tensor<f32>
// CHECK: %[[TEMP_101:.*]] = mhlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_102:.*]] = mhlo.multiply %[[TEMP_100]], %[[TEMP_101]] : tensor<f32>
// CHECK: %[[TEMP_103:.*]] = mhlo.compare  LT, %[[TEMP_4]], %[[TEMP_102]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_104:.*]] = mhlo.compare  LT, %[[TEMP_2]], %[[TEMP_11]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_105:.*]] = mhlo.and %[[TEMP_103]], %[[TEMP_104]] : tensor<i1>
// CHECK: %[[TEMP_106:.*]] = mhlo.multiply %[[TEMP_14]], %[[TEMP_36]] : tensor<f32>
// CHECK: %[[TEMP_107:.*]] = mhlo.add %[[TEMP_56]], %[[TEMP_11]] : tensor<f32>
// CHECK: %[[TEMP_108:.*]] = mhlo.divide %[[TEMP_106]], %[[TEMP_107]] : tensor<f32>
// CHECK: %[[TEMP_109:.*]] = mhlo.negate %[[TEMP_108]] : tensor<f32>
// CHECK: %[[TEMP_110:.*]] = mhlo.compare  GE, %[[TEMP_2]], %[[TEMP_11]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_111:.*]] = mhlo.multiply %[[TEMP_13]], %[[TEMP_59]] : tensor<f32>
// CHECK: %[[TEMP_112:.*]] = mhlo.divide %[[TEMP_111]], %[[TEMP_60]] : tensor<f32>
// CHECK: %[[TEMP_113:.*]] = mhlo.multiply %[[TEMP_13]], %[[TEMP_67]] : tensor<f32>
// CHECK: %[[TEMP_114:.*]] = mhlo.add %[[TEMP_112]], %[[TEMP_113]] : tensor<f32>
// CHECK: %[[TEMP_115:.*]] = mhlo.constant dense<1.500000e+00> : tensor<f32>
// CHECK: %[[TEMP_116:.*]] = mhlo.compare  LE, %[[TEMP_56]], %[[TEMP_115]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_117:.*]] = mhlo.divide %[[TEMP_111]], %[[TEMP_62]] : tensor<f32>
// CHECK: %[[TEMP_118:.*]] = mhlo.add %[[TEMP_112]], %[[TEMP_117]] : tensor<f32>
// CHECK: %[[TEMP_119:.*]] = mhlo.subtract %[[TEMP_56]], %[[TEMP_11]] : tensor<f32>
// CHECK: %[[TEMP_120:.*]] = mhlo.select %[[TEMP_116]], %[[TEMP_118]], %[[TEMP_119]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_121:.*]] = mhlo.select %[[TEMP_110]], %[[TEMP_114]], %[[TEMP_120]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_122:.*]] = mhlo.select %[[TEMP_105]], %[[TEMP_109]], %[[TEMP_121]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_123:.*]] = mhlo.multiply %[[TEMP_122]], %[[TEMP_107]] : tensor<f32>
// CHECK: %[[TEMP_124:.*]] = mhlo.sqrt %[[TEMP_123]] : tensor<f32>
// CHECK: %[[TEMP_125:.*]] = mhlo.divide %[[TEMP_4]], %[[TEMP_124]] : tensor<f32>
// CHECK: %[[TEMP_126:.*]] = mhlo.add %[[TEMP_122]], %[[TEMP_124]] : tensor<f32>
// CHECK: %[[TEMP_127:.*]] = mhlo.log_plus_one %[[TEMP_126]] : tensor<f32>
// CHECK: %[[TEMP_128:.*]] = mhlo.select %[[TEMP_105]], %[[TEMP_125]], %[[TEMP_127]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_129:.*]] = mhlo.select %[[TEMP_85]], %[[TEMP_98]], %[[TEMP_128]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_130:.*]] = mhlo.complex %[[TEMP_73]], %[[TEMP_129]] : tensor<complex<f32>>
// CHECK: %[[TEMP_131:.*]] = mhlo.real %[[TEMP_130]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_132:.*]] = mhlo.atan2 %[[TEMP_0]], %[[TEMP_131]] : tensor<f32>
// CHECK: %[[TEMP_133:.*]] = mhlo.imag %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_134:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_135:.*]] = mhlo.compare  LT, %[[TEMP_133]], %[[TEMP_134]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_136:.*]] = mhlo.imag %[[TEMP_130]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_137:.*]] = mhlo.negate %[[TEMP_136]] : tensor<f32>
// CHECK: %[[TEMP_138:.*]] = mhlo.select %[[TEMP_135]], %[[TEMP_137]], %[[TEMP_136]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_139:.*]] = mhlo.complex %[[TEMP_132]], %[[TEMP_138]] : tensor<complex<f32>>
// CHECK: return %[[TEMP_139]] : tensor<complex<f32>>
func.func @asin_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "chlo.asin"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// CHECK-LABEL: func.func @asin_complex_f64_dynamic(
// CHECK-SAME:   %[[TEMP_arg0:.*]]: tensor<?xcomplex<f64>>) -> tensor<?xcomplex<f64>> {
// CHECK: %[[TEMP_0:.*]] = mhlo.real %[[TEMP_arg0:.*]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK: %[[TEMP_1:.*]] = mhlo.real %[[TEMP_arg0:.*]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK: %[[TEMP_2:.*]] = mhlo.abs %[[TEMP_1]] : tensor<?xf64>
// CHECK: %[[TEMP_3:.*]] = mhlo.imag %[[TEMP_arg0:.*]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK: %[[TEMP_4:.*]] = mhlo.abs %[[TEMP_3]] : tensor<?xf64>
// CHECK: %[[TEMP_5:.*]] = mhlo.maximum %[[TEMP_2]], %[[TEMP_4]] : tensor<?xf64>
// CHECK: %[[TEMP_6:.*]] = mhlo.constant dense<1.7976931348623157E+308> : tensor<f64>
// CHECK: %[[TEMP_7:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_8:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_6]], %[[TEMP_7]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_9:.*]] = mhlo.sqrt %[[TEMP_8]] : tensor<?xf64>
// CHECK: %[[TEMP_10:.*]] = mhlo.constant dense<8.000000e+00> : tensor<f64>
// CHECK: %[[TEMP_11:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_12:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_10]], %[[TEMP_11]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_13:.*]] = mhlo.divide %[[TEMP_9]], %[[TEMP_12]] : tensor<?xf64>
// CHECK: %[[TEMP_14:.*]] = mhlo.compare  GE, %[[TEMP_5]], %[[TEMP_13]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_15:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK: %[[TEMP_16:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_17:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_15]], %[[TEMP_16]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_18:.*]] = mhlo.compare  LE, %[[TEMP_2]], %[[TEMP_17]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_19:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK: %[[TEMP_20:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_21:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_19]], %[[TEMP_20]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_22:.*]] = mhlo.add %[[TEMP_2]], %[[TEMP_17]] : tensor<?xf64>
// CHECK: %[[TEMP_23:.*]] = mhlo.abs %[[TEMP_22]] : tensor<?xf64>
// CHECK: %[[TEMP_24:.*]] = mhlo.maximum %[[TEMP_23]], %[[TEMP_4]] : tensor<?xf64>
// CHECK: %[[TEMP_25:.*]] = mhlo.minimum %[[TEMP_23]], %[[TEMP_4]] : tensor<?xf64>
// CHECK: %[[TEMP_26:.*]] = mhlo.compare  EQ, %[[TEMP_24]], %[[TEMP_25]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_27:.*]] = mhlo.constant dense<1.4142135623730951> : tensor<f64>
// CHECK: %[[TEMP_28:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_29:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_27]], %[[TEMP_28]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_30:.*]] = mhlo.multiply %[[TEMP_29]], %[[TEMP_24]] : tensor<?xf64>
// CHECK: %[[TEMP_31:.*]] = mhlo.divide %[[TEMP_25]], %[[TEMP_24]] : tensor<?xf64>
// CHECK: %[[TEMP_32:.*]] = mhlo.multiply %[[TEMP_31]], %[[TEMP_31]] : tensor<?xf64>
// CHECK: %[[TEMP_33:.*]] = mhlo.add %[[TEMP_17]], %[[TEMP_32]] : tensor<?xf64>
// CHECK: %[[TEMP_34:.*]] = mhlo.sqrt %[[TEMP_33]] : tensor<?xf64>
// CHECK: %[[TEMP_35:.*]] = mhlo.compare  EQ, %[[TEMP_34]], %[[TEMP_17]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_36:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[TEMP_37:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_38:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_36]], %[[TEMP_37]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_39:.*]] = mhlo.compare  GT, %[[TEMP_32]], %[[TEMP_38]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_40:.*]] = mhlo.and %[[TEMP_35]], %[[TEMP_39]] : tensor<?xi1>
// CHECK: %[[TEMP_41:.*]] = mhlo.multiply %[[TEMP_24]], %[[TEMP_32]] : tensor<?xf64>
// CHECK: %[[TEMP_42:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK: %[[TEMP_43:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_44:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_42]], %[[TEMP_43]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_45:.*]] = mhlo.divide %[[TEMP_41]], %[[TEMP_44]] : tensor<?xf64>
// CHECK: %[[TEMP_46:.*]] = mhlo.add %[[TEMP_24]], %[[TEMP_45]] : tensor<?xf64>
// CHECK: %[[TEMP_47:.*]] = mhlo.multiply %[[TEMP_24]], %[[TEMP_34]] : tensor<?xf64>
// CHECK: %[[TEMP_48:.*]] = mhlo.select %[[TEMP_40]], %[[TEMP_46]], %[[TEMP_47]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_49:.*]] = mhlo.select %[[TEMP_26]], %[[TEMP_30]], %[[TEMP_48]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_50:.*]] = mhlo.subtract %[[TEMP_2]], %[[TEMP_17]] : tensor<?xf64>
// CHECK: %[[TEMP_51:.*]] = mhlo.abs %[[TEMP_50]] : tensor<?xf64>
// CHECK: %[[TEMP_52:.*]] = mhlo.maximum %[[TEMP_51]], %[[TEMP_4]] : tensor<?xf64>
// CHECK: %[[TEMP_53:.*]] = mhlo.minimum %[[TEMP_51]], %[[TEMP_4]] : tensor<?xf64>
// CHECK: %[[TEMP_54:.*]] = mhlo.compare  EQ, %[[TEMP_52]], %[[TEMP_53]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_55:.*]] = mhlo.multiply %[[TEMP_29]], %[[TEMP_52]] : tensor<?xf64>
// CHECK: %[[TEMP_56:.*]] = mhlo.divide %[[TEMP_53]], %[[TEMP_52]] : tensor<?xf64>
// CHECK: %[[TEMP_57:.*]] = mhlo.multiply %[[TEMP_56]], %[[TEMP_56]] : tensor<?xf64>
// CHECK: %[[TEMP_58:.*]] = mhlo.add %[[TEMP_17]], %[[TEMP_57]] : tensor<?xf64>
// CHECK: %[[TEMP_59:.*]] = mhlo.sqrt %[[TEMP_58]] : tensor<?xf64>
// CHECK: %[[TEMP_60:.*]] = mhlo.compare  EQ, %[[TEMP_59]], %[[TEMP_17]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_61:.*]] = mhlo.compare  GT, %[[TEMP_57]], %[[TEMP_38]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_62:.*]] = mhlo.and %[[TEMP_60]], %[[TEMP_61]] : tensor<?xi1>
// CHECK: %[[TEMP_63:.*]] = mhlo.multiply %[[TEMP_52]], %[[TEMP_57]] : tensor<?xf64>
// CHECK: %[[TEMP_64:.*]] = mhlo.divide %[[TEMP_63]], %[[TEMP_44]] : tensor<?xf64>
// CHECK: %[[TEMP_65:.*]] = mhlo.add %[[TEMP_52]], %[[TEMP_64]] : tensor<?xf64>
// CHECK: %[[TEMP_66:.*]] = mhlo.multiply %[[TEMP_52]], %[[TEMP_59]] : tensor<?xf64>
// CHECK: %[[TEMP_67:.*]] = mhlo.select %[[TEMP_62]], %[[TEMP_65]], %[[TEMP_66]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_68:.*]] = mhlo.select %[[TEMP_54]], %[[TEMP_55]], %[[TEMP_67]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_69:.*]] = mhlo.add %[[TEMP_49]], %[[TEMP_68]] : tensor<?xf64>
// CHECK: %[[TEMP_70:.*]] = mhlo.multiply %[[TEMP_21]], %[[TEMP_69]] : tensor<?xf64>
// CHECK: %[[TEMP_71:.*]] = mhlo.add %[[TEMP_70]], %[[TEMP_2]] : tensor<?xf64>
// CHECK: %[[TEMP_72:.*]] = mhlo.multiply %[[TEMP_21]], %[[TEMP_71]] : tensor<?xf64>
// CHECK: %[[TEMP_73:.*]] = mhlo.multiply %[[TEMP_4]], %[[TEMP_4]] : tensor<?xf64>
// CHECK: %[[TEMP_74:.*]] = mhlo.add %[[TEMP_49]], %[[TEMP_22]] : tensor<?xf64>
// CHECK: %[[TEMP_75:.*]] = mhlo.divide %[[TEMP_73]], %[[TEMP_74]] : tensor<?xf64>
// CHECK: %[[TEMP_76:.*]] = mhlo.subtract %[[TEMP_68]], %[[TEMP_50]] : tensor<?xf64>
// CHECK: %[[TEMP_77:.*]] = mhlo.add %[[TEMP_75]], %[[TEMP_76]] : tensor<?xf64>
// CHECK: %[[TEMP_78:.*]] = mhlo.multiply %[[TEMP_72]], %[[TEMP_77]] : tensor<?xf64>
// CHECK: %[[TEMP_79:.*]] = mhlo.sqrt %[[TEMP_78]] : tensor<?xf64>
// CHECK: %[[TEMP_80:.*]] = mhlo.divide %[[TEMP_72]], %[[TEMP_74]] : tensor<?xf64>
// CHECK: %[[TEMP_81:.*]] = mhlo.add %[[TEMP_68]], %[[TEMP_50]] : tensor<?xf64>
// CHECK: %[[TEMP_82:.*]] = mhlo.divide %[[TEMP_72]], %[[TEMP_81]] : tensor<?xf64>
// CHECK: %[[TEMP_83:.*]] = mhlo.add %[[TEMP_80]], %[[TEMP_82]] : tensor<?xf64>
// CHECK: %[[TEMP_84:.*]] = mhlo.sqrt %[[TEMP_83]] : tensor<?xf64>
// CHECK: %[[TEMP_85:.*]] = mhlo.multiply %[[TEMP_4]], %[[TEMP_84]] : tensor<?xf64>
// CHECK: %[[TEMP_86:.*]] = mhlo.select %[[TEMP_18]], %[[TEMP_79]], %[[TEMP_85]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_87:.*]] = mhlo.select %[[TEMP_14]], %[[TEMP_4]], %[[TEMP_86]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_88:.*]] = mhlo.constant dense<1.000000e+12> : tensor<f64>
// CHECK: %[[TEMP_89:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_90:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_88]], %[[TEMP_89]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_91:.*]] = mhlo.multiply %[[TEMP_13]], %[[TEMP_90]] : tensor<?xf64>
// CHECK: %[[TEMP_92:.*]] = mhlo.compare  LT, %[[TEMP_2]], %[[TEMP_91]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_93:.*]] = mhlo.constant dense<9.9999999999999995E-7> : tensor<f64>
// CHECK: %[[TEMP_94:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_95:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_93]], %[[TEMP_94]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_96:.*]] = mhlo.multiply %[[TEMP_13]], %[[TEMP_95]] : tensor<?xf64>
// CHECK: %[[TEMP_97:.*]] = mhlo.constant dense<1.000000e+02> : tensor<f64>
// CHECK: %[[TEMP_98:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_99:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_97]], %[[TEMP_98]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_100:.*]] = mhlo.multiply %[[TEMP_13]], %[[TEMP_99]] : tensor<?xf64>
// CHECK: %[[TEMP_101:.*]] = mhlo.select %[[TEMP_92]], %[[TEMP_96]], %[[TEMP_100]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_102:.*]] = mhlo.compare  GE, %[[TEMP_4]], %[[TEMP_101]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_103:.*]] = mhlo.select %[[TEMP_102]], %[[TEMP_4]], %[[TEMP_2]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_104:.*]] = mhlo.select %[[TEMP_102]], %[[TEMP_101]], %[[TEMP_13]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_105:.*]] = mhlo.compare  GE, %[[TEMP_103]], %[[TEMP_104]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_106:.*]] = mhlo.log %[[TEMP_44]] : tensor<?xf64>
// CHECK: %[[TEMP_107:.*]] = mhlo.log %[[TEMP_103]] : tensor<?xf64>
// CHECK: %[[TEMP_108:.*]] = mhlo.add %[[TEMP_106]], %[[TEMP_107]] : tensor<?xf64>
// CHECK: %[[TEMP_109:.*]] = mhlo.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK: %[[TEMP_110:.*]] = shape.shape_of %[[TEMP_3]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_111:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_109]], %[[TEMP_110]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_112:.*]] = mhlo.compare  EQ, %[[TEMP_4]], %[[TEMP_111]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_113:.*]] = mhlo.not %[[TEMP_112]] : tensor<?xi1>
// CHECK: %[[TEMP_114:.*]] = mhlo.and %[[TEMP_102]], %[[TEMP_113]] : tensor<?xi1>
// CHECK: %[[TEMP_115:.*]] = mhlo.divide %[[TEMP_2]], %[[TEMP_4]] : tensor<?xf64>
// CHECK: %[[TEMP_116:.*]] = mhlo.select %[[TEMP_114]], %[[TEMP_115]], %[[TEMP_38]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_117:.*]] = mhlo.multiply %[[TEMP_116]], %[[TEMP_116]] : tensor<?xf64>
// CHECK: %[[TEMP_118:.*]] = mhlo.log_plus_one %[[TEMP_117]] : tensor<?xf64>
// CHECK: %[[TEMP_119:.*]] = mhlo.multiply %[[TEMP_21]], %[[TEMP_118]] : tensor<?xf64>
// CHECK: %[[TEMP_120:.*]] = mhlo.add %[[TEMP_108]], %[[TEMP_119]] : tensor<?xf64>
// CHECK: %[[TEMP_121:.*]] = mhlo.constant dense<2.2250738585072014E-308> : tensor<f64>
// CHECK: %[[TEMP_122:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_123:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_121]], %[[TEMP_122]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_124:.*]] = mhlo.sqrt %[[TEMP_123]] : tensor<?xf64>
// CHECK: %[[TEMP_125:.*]] = mhlo.constant dense<4.000000e+00> : tensor<f64>
// CHECK: %[[TEMP_126:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_127:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_125]], %[[TEMP_126]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_128:.*]] = mhlo.multiply %[[TEMP_124]], %[[TEMP_127]] : tensor<?xf64>
// CHECK: %[[TEMP_129:.*]] = mhlo.compare  LT, %[[TEMP_4]], %[[TEMP_128]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_130:.*]] = mhlo.compare  LT, %[[TEMP_2]], %[[TEMP_17]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_131:.*]] = mhlo.and %[[TEMP_129]], %[[TEMP_130]] : tensor<?xi1>
// CHECK: %[[TEMP_132:.*]] = mhlo.multiply %[[TEMP_22]], %[[TEMP_50]] : tensor<?xf64>
// CHECK: %[[TEMP_133:.*]] = mhlo.add %[[TEMP_70]], %[[TEMP_17]] : tensor<?xf64>
// CHECK: %[[TEMP_134:.*]] = mhlo.divide %[[TEMP_132]], %[[TEMP_133]] : tensor<?xf64>
// CHECK: %[[TEMP_135:.*]] = mhlo.negate %[[TEMP_134]] : tensor<?xf64>
// CHECK: %[[TEMP_136:.*]] = mhlo.compare  GE, %[[TEMP_2]], %[[TEMP_17]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_137:.*]] = mhlo.multiply %[[TEMP_21]], %[[TEMP_73]] : tensor<?xf64>
// CHECK: %[[TEMP_138:.*]] = mhlo.divide %[[TEMP_137]], %[[TEMP_74]] : tensor<?xf64>
// CHECK: %[[TEMP_139:.*]] = mhlo.multiply %[[TEMP_21]], %[[TEMP_81]] : tensor<?xf64>
// CHECK: %[[TEMP_140:.*]] = mhlo.add %[[TEMP_138]], %[[TEMP_139]] : tensor<?xf64>
// CHECK: %[[TEMP_141:.*]] = mhlo.constant dense<1.500000e+00> : tensor<f64>
// CHECK: %[[TEMP_142:.*]] = shape.shape_of %[[TEMP_1]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_143:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_141]], %[[TEMP_142]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_144:.*]] = mhlo.compare  LE, %[[TEMP_70]], %[[TEMP_143]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_145:.*]] = mhlo.divide %[[TEMP_137]], %[[TEMP_76]] : tensor<?xf64>
// CHECK: %[[TEMP_146:.*]] = mhlo.add %[[TEMP_138]], %[[TEMP_145]] : tensor<?xf64>
// CHECK: %[[TEMP_147:.*]] = mhlo.subtract %[[TEMP_70]], %[[TEMP_17]] : tensor<?xf64>
// CHECK: %[[TEMP_148:.*]] = mhlo.select %[[TEMP_144]], %[[TEMP_146]], %[[TEMP_147]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_149:.*]] = mhlo.select %[[TEMP_136]], %[[TEMP_140]], %[[TEMP_148]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_150:.*]] = mhlo.select %[[TEMP_131]], %[[TEMP_135]], %[[TEMP_149]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_151:.*]] = mhlo.multiply %[[TEMP_150]], %[[TEMP_133]] : tensor<?xf64>
// CHECK: %[[TEMP_152:.*]] = mhlo.sqrt %[[TEMP_151]] : tensor<?xf64>
// CHECK: %[[TEMP_153:.*]] = mhlo.divide %[[TEMP_4]], %[[TEMP_152]] : tensor<?xf64>
// CHECK: %[[TEMP_154:.*]] = mhlo.add %[[TEMP_150]], %[[TEMP_152]] : tensor<?xf64>
// CHECK: %[[TEMP_155:.*]] = mhlo.log_plus_one %[[TEMP_154]] : tensor<?xf64>
// CHECK: %[[TEMP_156:.*]] = mhlo.select %[[TEMP_131]], %[[TEMP_153]], %[[TEMP_155]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_157:.*]] = mhlo.select %[[TEMP_105]], %[[TEMP_120]], %[[TEMP_156]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_158:.*]] = mhlo.complex %[[TEMP_87]], %[[TEMP_157]] : tensor<?xcomplex<f64>>
// CHECK: %[[TEMP_159:.*]] = mhlo.real %[[TEMP_158]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK: %[[TEMP_160:.*]] = mhlo.atan2 %[[TEMP_0]], %[[TEMP_159]] : tensor<?xf64>
// CHECK: %[[TEMP_161:.*]] = mhlo.imag %[[TEMP_arg0:.*]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK: %[[TEMP_162:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[TEMP_163:.*]] = shape.shape_of %[[TEMP_0]] : tensor<?xf64> -> tensor<1xindex>
// CHECK: %[[TEMP_164:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TEMP_162]], %[[TEMP_163]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f64>, tensor<1xindex>) -> tensor<?xf64>
// CHECK: %[[TEMP_165:.*]] = mhlo.compare  LT, %[[TEMP_161]], %[[TEMP_164]] : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
// CHECK: %[[TEMP_166:.*]] = mhlo.imag %[[TEMP_158]] : (tensor<?xcomplex<f64>>) -> tensor<?xf64>
// CHECK: %[[TEMP_167:.*]] = mhlo.negate %[[TEMP_166]] : tensor<?xf64>
// CHECK: %[[TEMP_168:.*]] = mhlo.select %[[TEMP_165]], %[[TEMP_167]], %[[TEMP_166]] : tensor<?xi1>, tensor<?xf64>
// CHECK: %[[TEMP_169:.*]] = mhlo.complex %[[TEMP_160]], %[[TEMP_168]] : tensor<?xcomplex<f64>>
// CHECK: return %[[TEMP_169]] : tensor<?xcomplex<f64>>
func.func @asin_complex_f64_dynamic(%arg : tensor<?xcomplex<f64>>) -> tensor<?xcomplex<f64>> {
  %result = "chlo.asin"(%arg) : (tensor<?xcomplex<f64>>) -> tensor<?xcomplex<f64>>
  func.return %result : tensor<?xcomplex<f64>>
}

// -----

// CHECK-LABEL: @asinh_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func.func @asinh_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  // Check for the bf16-specific max value.
  // CHECK: mhlo.constant dense<3.389{{.*}}e+38>
  %result = "chlo.asinh"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %result : tensor<bf16>
}

// -----

// CHECK-LABEL: @asinh_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func.func @asinh_f16(%arg : tensor<f16>) -> tensor<f16> {
  // Check for the f16-specific max value.
  // CHECK: mhlo.constant dense<6.550{{.*}}e+04>
  %result = "chlo.asinh"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %result : tensor<f16>
}

// -----

// CHECK-LABEL: @asinh_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @asinh_f32(%arg : tensor<f32>) -> tensor<f32> {
  // Check for the f32-specific max value.
  // CHECK: mhlo.constant dense<3.402{{.*}}E+38>
  %result = "chlo.asinh"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// -----

// CHECK-LABEL:  @asinh_f64
// CHECK-SAME:   %[[VAL_0:.*]]: tensor<f64>) -> tensor<f64> {
func.func @asinh_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK:   %[[VAL_1:.*]] = mhlo.sign %[[VAL_0]] : tensor<f64>
  // CHECK:   %[[VAL_2:.*]] = mhlo.abs %[[VAL_0]] : tensor<f64>
  // CHECK:   %[[VAL_3:.*]] = mhlo.constant dense<1.7976931348623157E+308> : tensor<f64>
  // CHECK:   %[[VAL_4:.*]] = mhlo.sqrt %[[VAL_3]] : tensor<f64>
  // CHECK:   %[[VAL_5:.*]] = mhlo.compare  GE, %[[VAL_2]], %[[VAL_4]] : (tensor<f64>, tensor<f64>) -> tensor<i1>
  // CHECK:   %[[VAL_6:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f64>
  // CHECK:   %[[VAL_7:.*]] = mhlo.log %[[VAL_6]] : tensor<f64>
  // CHECK:   %[[VAL_8:.*]] = mhlo.log %[[VAL_2]] : tensor<f64>
  // CHECK:   %[[VAL_9:.*]] = mhlo.add %[[VAL_7]], %[[VAL_8]] : tensor<f64>
  // CHECK:   %[[VAL_10:.*]] = mhlo.multiply %[[VAL_2]], %[[VAL_2]] : tensor<f64>
  // CHECK:   %[[VAL_11:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f64>
  // CHECK:   %[[VAL_12:.*]] = mhlo.add %[[VAL_11]], %[[VAL_10]] : tensor<f64>
  // CHECK:   %[[VAL_13:.*]] = mhlo.sqrt %[[VAL_12]] : tensor<f64>
  // CHECK:   %[[VAL_14:.*]] = mhlo.add %[[VAL_11]], %[[VAL_13]] : tensor<f64>
  // CHECK:   %[[VAL_15:.*]] = mhlo.divide %[[VAL_10]], %[[VAL_14]] : tensor<f64>
  // CHECK:   %[[VAL_16:.*]] = mhlo.add %[[VAL_2]], %[[VAL_15]] : tensor<f64>
  // CHECK:   %[[VAL_17:.*]] = mhlo.log_plus_one %[[VAL_16]] : tensor<f64>
  // CHECK:   %[[VAL_18:.*]] = mhlo.select %[[VAL_5]], %[[VAL_9]], %[[VAL_17]] : tensor<i1>, tensor<f64>
  // CHECK:   %[[VAL_19:.*]] = mhlo.multiply %[[VAL_1]], %[[VAL_18]] : tensor<f64>
  // CHECK:   return %[[VAL_19]] : tensor<f64>
  // CHECK: }
  %result = "chlo.asinh"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %result : tensor<f64>
}

// -----

// CHECK-LABEL: func.func @acosh_complex_f32(
// CHECK-SAME:   %[[TEMP_arg0:.*]]: tensor<complex<f32>>) -> tensor<complex<f32>> {
// CHECK: %[[TEMP_0:.*]] = mhlo.real %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_1:.*]] = mhlo.abs %[[TEMP_0]] : tensor<f32>
// CHECK: %[[TEMP_2:.*]] = mhlo.imag %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_3:.*]] = mhlo.abs %[[TEMP_2]] : tensor<f32>
// CHECK: %[[TEMP_4:.*]] = mhlo.maximum %[[TEMP_1]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_5:.*]] = mhlo.constant dense<3.40282347E+38> : tensor<f32>
// CHECK: %[[TEMP_6:.*]] = mhlo.sqrt %[[TEMP_5]] : tensor<f32>
// CHECK: %[[TEMP_7:.*]] = mhlo.constant dense<8.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_8:.*]] = mhlo.divide %[[TEMP_6]], %[[TEMP_7]] : tensor<f32>
// CHECK: %[[TEMP_9:.*]] = mhlo.compare  GE, %[[TEMP_4]], %[[TEMP_8]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_10:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_11:.*]] = mhlo.compare  LE, %[[TEMP_1]], %[[TEMP_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_12:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK: %[[TEMP_13:.*]] = mhlo.add %[[TEMP_1]], %[[TEMP_10]] : tensor<f32>
// CHECK: %[[TEMP_14:.*]] = mhlo.abs %[[TEMP_13]] : tensor<f32>
// CHECK: %[[TEMP_15:.*]] = mhlo.maximum %[[TEMP_14]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_16:.*]] = mhlo.minimum %[[TEMP_14]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_17:.*]] = mhlo.compare  EQ, %[[TEMP_15]], %[[TEMP_16]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_18:.*]] = mhlo.constant dense<1.41421354> : tensor<f32>
// CHECK: %[[TEMP_19:.*]] = mhlo.multiply %[[TEMP_18]], %[[TEMP_15]] : tensor<f32>
// CHECK: %[[TEMP_20:.*]] = mhlo.divide %[[TEMP_16]], %[[TEMP_15]] : tensor<f32>
// CHECK: %[[TEMP_21:.*]] = mhlo.multiply %[[TEMP_20]], %[[TEMP_20]] : tensor<f32>
// CHECK: %[[TEMP_22:.*]] = mhlo.add %[[TEMP_10]], %[[TEMP_21]] : tensor<f32>
// CHECK: %[[TEMP_23:.*]] = mhlo.sqrt %[[TEMP_22]] : tensor<f32>
// CHECK: %[[TEMP_24:.*]] = mhlo.compare  EQ, %[[TEMP_23]], %[[TEMP_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_25:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_26:.*]] = mhlo.compare  GT, %[[TEMP_21]], %[[TEMP_25]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_27:.*]] = mhlo.and %[[TEMP_24]], %[[TEMP_26]] : tensor<i1>
// CHECK: %[[TEMP_28:.*]] = mhlo.multiply %[[TEMP_15]], %[[TEMP_21]] : tensor<f32>
// CHECK: %[[TEMP_29:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_30:.*]] = mhlo.divide %[[TEMP_28]], %[[TEMP_29]] : tensor<f32>
// CHECK: %[[TEMP_31:.*]] = mhlo.add %[[TEMP_15]], %[[TEMP_30]] : tensor<f32>
// CHECK: %[[TEMP_32:.*]] = mhlo.multiply %[[TEMP_15]], %[[TEMP_23]] : tensor<f32>
// CHECK: %[[TEMP_33:.*]] = mhlo.select %[[TEMP_27]], %[[TEMP_31]], %[[TEMP_32]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_34:.*]] = mhlo.select %[[TEMP_17]], %[[TEMP_19]], %[[TEMP_33]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_35:.*]] = mhlo.subtract %[[TEMP_1]], %[[TEMP_10]] : tensor<f32>
// CHECK: %[[TEMP_36:.*]] = mhlo.abs %[[TEMP_35]] : tensor<f32>
// CHECK: %[[TEMP_37:.*]] = mhlo.maximum %[[TEMP_36]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_38:.*]] = mhlo.minimum %[[TEMP_36]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_39:.*]] = mhlo.compare  EQ, %[[TEMP_37]], %[[TEMP_38]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_40:.*]] = mhlo.multiply %[[TEMP_18]], %[[TEMP_37]] : tensor<f32>
// CHECK: %[[TEMP_41:.*]] = mhlo.divide %[[TEMP_38]], %[[TEMP_37]] : tensor<f32>
// CHECK: %[[TEMP_42:.*]] = mhlo.multiply %[[TEMP_41]], %[[TEMP_41]] : tensor<f32>
// CHECK: %[[TEMP_43:.*]] = mhlo.add %[[TEMP_10]], %[[TEMP_42]] : tensor<f32>
// CHECK: %[[TEMP_44:.*]] = mhlo.sqrt %[[TEMP_43]] : tensor<f32>
// CHECK: %[[TEMP_45:.*]] = mhlo.compare  EQ, %[[TEMP_44]], %[[TEMP_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_46:.*]] = mhlo.compare  GT, %[[TEMP_42]], %[[TEMP_25]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_47:.*]] = mhlo.and %[[TEMP_45]], %[[TEMP_46]] : tensor<i1>
// CHECK: %[[TEMP_48:.*]] = mhlo.multiply %[[TEMP_37]], %[[TEMP_42]] : tensor<f32>
// CHECK: %[[TEMP_49:.*]] = mhlo.divide %[[TEMP_48]], %[[TEMP_29]] : tensor<f32>
// CHECK: %[[TEMP_50:.*]] = mhlo.add %[[TEMP_37]], %[[TEMP_49]] : tensor<f32>
// CHECK: %[[TEMP_51:.*]] = mhlo.multiply %[[TEMP_37]], %[[TEMP_44]] : tensor<f32>
// CHECK: %[[TEMP_52:.*]] = mhlo.select %[[TEMP_47]], %[[TEMP_50]], %[[TEMP_51]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_53:.*]] = mhlo.select %[[TEMP_39]], %[[TEMP_40]], %[[TEMP_52]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_54:.*]] = mhlo.add %[[TEMP_34]], %[[TEMP_53]] : tensor<f32>
// CHECK: %[[TEMP_55:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_54]] : tensor<f32>
// CHECK: %[[TEMP_56:.*]] = mhlo.add %[[TEMP_55]], %[[TEMP_1]] : tensor<f32>
// CHECK: %[[TEMP_57:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_56]] : tensor<f32>
// CHECK: %[[TEMP_58:.*]] = mhlo.multiply %[[TEMP_3]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_59:.*]] = mhlo.add %[[TEMP_34]], %[[TEMP_13]] : tensor<f32>
// CHECK: %[[TEMP_60:.*]] = mhlo.divide %[[TEMP_58]], %[[TEMP_59]] : tensor<f32>
// CHECK: %[[TEMP_61:.*]] = mhlo.subtract %[[TEMP_53]], %[[TEMP_35]] : tensor<f32>
// CHECK: %[[TEMP_62:.*]] = mhlo.add %[[TEMP_60]], %[[TEMP_61]] : tensor<f32>
// CHECK: %[[TEMP_63:.*]] = mhlo.multiply %[[TEMP_57]], %[[TEMP_62]] : tensor<f32>
// CHECK: %[[TEMP_64:.*]] = mhlo.sqrt %[[TEMP_63]] : tensor<f32>
// CHECK: %[[TEMP_65:.*]] = mhlo.divide %[[TEMP_57]], %[[TEMP_59]] : tensor<f32>
// CHECK: %[[TEMP_66:.*]] = mhlo.add %[[TEMP_53]], %[[TEMP_35]] : tensor<f32>
// CHECK: %[[TEMP_67:.*]] = mhlo.divide %[[TEMP_57]], %[[TEMP_66]] : tensor<f32>
// CHECK: %[[TEMP_68:.*]] = mhlo.add %[[TEMP_65]], %[[TEMP_67]] : tensor<f32>
// CHECK: %[[TEMP_69:.*]] = mhlo.sqrt %[[TEMP_68]] : tensor<f32>
// CHECK: %[[TEMP_70:.*]] = mhlo.multiply %[[TEMP_3]], %[[TEMP_69]] : tensor<f32>
// CHECK: %[[TEMP_71:.*]] = mhlo.select %[[TEMP_11]], %[[TEMP_64]], %[[TEMP_70]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_72:.*]] = mhlo.select %[[TEMP_9]], %[[TEMP_3]], %[[TEMP_71]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_73:.*]] = mhlo.constant dense<9.99999995E+11> : tensor<f32>
// CHECK: %[[TEMP_74:.*]] = mhlo.multiply %[[TEMP_8]], %[[TEMP_73]] : tensor<f32>
// CHECK: %[[TEMP_75:.*]] = mhlo.compare  LT, %[[TEMP_1]], %[[TEMP_74]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_76:.*]] = mhlo.constant dense<9.99999997E-7> : tensor<f32>
// CHECK: %[[TEMP_77:.*]] = mhlo.multiply %[[TEMP_8]], %[[TEMP_76]] : tensor<f32>
// CHECK: %[[TEMP_78:.*]] = mhlo.constant dense<1.000000e+02> : tensor<f32>
// CHECK: %[[TEMP_79:.*]] = mhlo.multiply %[[TEMP_8]], %[[TEMP_78]] : tensor<f32>
// CHECK: %[[TEMP_80:.*]] = mhlo.select %[[TEMP_75]], %[[TEMP_77]], %[[TEMP_79]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_81:.*]] = mhlo.compare  GE, %[[TEMP_3]], %[[TEMP_80]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_82:.*]] = mhlo.select %[[TEMP_81]], %[[TEMP_3]], %[[TEMP_1]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_83:.*]] = mhlo.select %[[TEMP_81]], %[[TEMP_80]], %[[TEMP_8]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_84:.*]] = mhlo.compare  GE, %[[TEMP_82]], %[[TEMP_83]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_85:.*]] = mhlo.log %[[TEMP_29]] : tensor<f32>
// CHECK: %[[TEMP_86:.*]] = mhlo.log %[[TEMP_82]] : tensor<f32>
// CHECK: %[[TEMP_87:.*]] = mhlo.add %[[TEMP_85]], %[[TEMP_86]] : tensor<f32>
// CHECK: %[[TEMP_88:.*]] = mhlo.constant dense<0x7F800000> : tensor<f32>
// CHECK: %[[TEMP_89:.*]] = mhlo.compare  EQ, %[[TEMP_3]], %[[TEMP_88]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_90:.*]] = mhlo.not %[[TEMP_89]] : tensor<i1>
// CHECK: %[[TEMP_91:.*]] = mhlo.and %[[TEMP_81]], %[[TEMP_90]] : tensor<i1>
// CHECK: %[[TEMP_92:.*]] = mhlo.divide %[[TEMP_1]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_93:.*]] = mhlo.select %[[TEMP_91]], %[[TEMP_92]], %[[TEMP_25]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_94:.*]] = mhlo.multiply %[[TEMP_93]], %[[TEMP_93]] : tensor<f32>
// CHECK: %[[TEMP_95:.*]] = mhlo.log_plus_one %[[TEMP_94]] : tensor<f32>
// CHECK: %[[TEMP_96:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_95]] : tensor<f32>
// CHECK: %[[TEMP_97:.*]] = mhlo.add %[[TEMP_87]], %[[TEMP_96]] : tensor<f32>
// CHECK: %[[TEMP_98:.*]] = mhlo.constant dense<1.17549435E-38> : tensor<f32>
// CHECK: %[[TEMP_99:.*]] = mhlo.sqrt %[[TEMP_98]] : tensor<f32>
// CHECK: %[[TEMP_100:.*]] = mhlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_101:.*]] = mhlo.multiply %[[TEMP_99]], %[[TEMP_100]] : tensor<f32>
// CHECK: %[[TEMP_102:.*]] = mhlo.compare  LT, %[[TEMP_3]], %[[TEMP_101]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_103:.*]] = mhlo.compare  LT, %[[TEMP_1]], %[[TEMP_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_104:.*]] = mhlo.and %[[TEMP_102]], %[[TEMP_103]] : tensor<i1>
// CHECK: %[[TEMP_105:.*]] = mhlo.multiply %[[TEMP_13]], %[[TEMP_35]] : tensor<f32>
// CHECK: %[[TEMP_106:.*]] = mhlo.add %[[TEMP_55]], %[[TEMP_10]] : tensor<f32>
// CHECK: %[[TEMP_107:.*]] = mhlo.divide %[[TEMP_105]], %[[TEMP_106]] : tensor<f32>
// CHECK: %[[TEMP_108:.*]] = mhlo.negate %[[TEMP_107]] : tensor<f32>
// CHECK: %[[TEMP_109:.*]] = mhlo.compare  GE, %[[TEMP_1]], %[[TEMP_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_110:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_58]] : tensor<f32>
// CHECK: %[[TEMP_111:.*]] = mhlo.divide %[[TEMP_110]], %[[TEMP_59]] : tensor<f32>
// CHECK: %[[TEMP_112:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_66]] : tensor<f32>
// CHECK: %[[TEMP_113:.*]] = mhlo.add %[[TEMP_111]], %[[TEMP_112]] : tensor<f32>
// CHECK: %[[TEMP_114:.*]] = mhlo.constant dense<1.500000e+00> : tensor<f32>
// CHECK: %[[TEMP_115:.*]] = mhlo.compare  LE, %[[TEMP_55]], %[[TEMP_114]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_116:.*]] = mhlo.divide %[[TEMP_110]], %[[TEMP_61]] : tensor<f32>
// CHECK: %[[TEMP_117:.*]] = mhlo.add %[[TEMP_111]], %[[TEMP_116]] : tensor<f32>
// CHECK: %[[TEMP_118:.*]] = mhlo.subtract %[[TEMP_55]], %[[TEMP_10]] : tensor<f32>
// CHECK: %[[TEMP_119:.*]] = mhlo.select %[[TEMP_115]], %[[TEMP_117]], %[[TEMP_118]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_120:.*]] = mhlo.select %[[TEMP_109]], %[[TEMP_113]], %[[TEMP_119]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_121:.*]] = mhlo.select %[[TEMP_104]], %[[TEMP_108]], %[[TEMP_120]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_122:.*]] = mhlo.multiply %[[TEMP_121]], %[[TEMP_106]] : tensor<f32>
// CHECK: %[[TEMP_123:.*]] = mhlo.sqrt %[[TEMP_122]] : tensor<f32>
// CHECK: %[[TEMP_124:.*]] = mhlo.divide %[[TEMP_3]], %[[TEMP_123]] : tensor<f32>
// CHECK: %[[TEMP_125:.*]] = mhlo.add %[[TEMP_121]], %[[TEMP_123]] : tensor<f32>
// CHECK: %[[TEMP_126:.*]] = mhlo.log_plus_one %[[TEMP_125]] : tensor<f32>
// CHECK: %[[TEMP_127:.*]] = mhlo.select %[[TEMP_104]], %[[TEMP_124]], %[[TEMP_126]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_128:.*]] = mhlo.select %[[TEMP_84]], %[[TEMP_97]], %[[TEMP_127]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_129:.*]] = mhlo.complex %[[TEMP_72]], %[[TEMP_128]] : tensor<complex<f32>>
// CHECK: %[[TEMP_130:.*]] = mhlo.imag %[[TEMP_129]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_131:.*]] = mhlo.imag %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_132:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_133:.*]] = mhlo.compare  LT, %[[TEMP_131]], %[[TEMP_132]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_134:.*]] = mhlo.real %[[TEMP_129]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_135:.*]] = mhlo.real %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_136:.*]] = mhlo.atan2 %[[TEMP_134]], %[[TEMP_135]] : tensor<f32>
// CHECK: %[[TEMP_137:.*]] = mhlo.negate %[[TEMP_136]] : tensor<f32>
// CHECK: %[[TEMP_138:.*]] = mhlo.select %[[TEMP_133]], %[[TEMP_137]], %[[TEMP_136]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_139:.*]] = mhlo.complex %[[TEMP_130]], %[[TEMP_138]] : tensor<complex<f32>>
// CHECK: return %[[TEMP_139]] : tensor<complex<f32>>
func.func @acosh_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "chlo.acosh"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// Lower statically shaped `constant_like` to constant.
// CHECK-LABEL: @constant_like_static_shape
func.func @constant_like_static_shape(%arg : tensor<1x2xi64>) -> tensor<1x2xf32> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant dense<3.200000e+00> : tensor<1x2xf32>
  // CHECK: return %[[RESULT]]
  %result = "chlo.constant_like"(%arg) { value = 3.2 : f32 }
      : (tensor<1x2xi64>) -> tensor<1x2xf32>
  func.return %result : tensor<1x2xf32>
}

// -----

// Lower dynamically shaped `constant_like` to broadcasted constant.
// CHECK-LABEL: constant_like_dynamic_shape
// CHECK-SAME: (%[[ARG:.*]]: tensor<?x?xi64>)
func.func @constant_like_dynamic_shape(%arg : tensor<?x?xi64>) -> tensor<?x?xf32> {
  // CHECK: %[[CONSTANT:.*]] = mhlo.constant dense<3.200000e+00> : tensor<f32>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG]] : tensor<?x?xi64> -> tensor<2xindex>
  // CHECK: %[[BROADCASTED_CONSTANT:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[CONSTANT]], %[[SHAPE]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK: return %[[BROADCASTED_CONSTANT]] : tensor<?x?xf32>
  %result = "chlo.constant_like"(%arg) { value = 3.2 : f32 }
      : (tensor<?x?xi64>) -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @conj
func.func @conj(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>> {
  // CHECK-SAME: ([[INPUT:%.*]]: tensor
  // CHECK-NEXT: [[R1:%.*]] = mhlo.real [[INPUT]]
  // CHECK-NEXT: [[R2:%.*]] = mhlo.imag [[INPUT]]
  // CHECK-NEXT: [[R3:%.*]] = mhlo.negate [[R2]]
  // CHECK-NEXT: [[R4:%.*]] = mhlo.complex [[R1]], [[R3]]
  %1 = "chlo.conj"(%arg0) : (tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>>
  func.return %1 : tensor<3xcomplex<f32>>
}

// -----

// CHECK-LABEL: @erf_f64
// CHECK-SAME: %[[ARG:.*]]: tensor<f64>
func.func @erf_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK-HIGH-LEVEL: mhlo.erf
  // CHECK: %[[RESULT:.*]] = mhlo.erf %[[ARG]]
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @erf_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @erf_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK-HIGH-LEVEL: mhlo.erf
  // CHECK: %[[RESULT:.*]] = mhlo.erf %[[ARG]]
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @erf_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func.func @erf_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK-HIGH-LEVEL: mhlo.erf
  // CHECK: %[[RESULT:.*]] = mhlo.erf %[[ARG]]
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @erf_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func.func @erf_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  // CHECK-HIGH-LEVEL: mhlo.erf
  // CHECK: %[[RESULT:.*]] = mhlo.erf %[[ARG]]
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %1 : tensor<bf16>
}

// -----

// CHECK-LABEL: @acosh
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<f16>) -> tensor<f16> {
func.func @acosh(%arg: tensor<f16>) -> tensor<f16> {
  // CHECK-DAG:   %[[VAL_1:.*]] = mhlo.constant dense<6.550400e+04> : tensor<f16>
  // CHECK-DAG:   %[[VAL_2:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f16>
  // CHECK:   %[[VAL_3:.*]] = mhlo.divide %[[VAL_1]], %[[VAL_2]] : tensor<f16>
  // CHECK:   %[[VAL_4:.*]] = mhlo.compare  GE, %[[VAL_0]], %[[VAL_3]] : (tensor<f16>, tensor<f16>) -> tensor<i1>
  // CHECK:   %[[VAL_5:.*]] = mhlo.log %[[VAL_2]] : tensor<f16>
  // CHECK:   %[[VAL_6:.*]] = mhlo.log %[[VAL_0]] : tensor<f16>
  // CHECK:   %[[VAL_7:.*]] = mhlo.add %[[VAL_5]], %[[VAL_6]] : tensor<f16>
  // CHECK:   %[[VAL_8:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f16>
  // CHECK:   %[[VAL_9:.*]] = mhlo.subtract %[[VAL_0]], %[[VAL_8]] : tensor<f16>
  // CHECK:   %[[VAL_10:.*]] = mhlo.sqrt %[[VAL_9]] : tensor<f16>
  // CHECK:   %[[VAL_11:.*]] = mhlo.add %[[VAL_0]], %[[VAL_8]] : tensor<f16>
  // CHECK:   %[[VAL_12:.*]] = mhlo.sqrt %[[VAL_11]] : tensor<f16>
  // CHECK:   %[[VAL_13:.*]] = mhlo.add %[[VAL_12]], %[[VAL_10]] : tensor<f16>
  // CHECK:   %[[VAL_14:.*]] = mhlo.multiply %[[VAL_10]], %[[VAL_13]] : tensor<f16>
  // CHECK:   %[[VAL_15:.*]] = mhlo.log_plus_one %[[VAL_14]] : tensor<f16>
  // CHECK:   %[[VAL_16:.*]] = mhlo.select %[[VAL_4]], %[[VAL_7]], %[[VAL_15]] : tensor<i1>, tensor<f16>
  // CHECK:   return %[[VAL_16]] : tensor<f16>
  // CHECK: }
  %1 = "chlo.acosh"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: func.func @acosh_complex_f32(
// CHECK-SAME:   %[[TEMP_arg0:.*]]: tensor<complex<f32>>) -> tensor<complex<f32>> {
// CHECK: %[[TEMP_0:.*]] = mhlo.real %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_1:.*]] = mhlo.abs %[[TEMP_0]] : tensor<f32>
// CHECK: %[[TEMP_2:.*]] = mhlo.imag %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_3:.*]] = mhlo.abs %[[TEMP_2]] : tensor<f32>
// CHECK: %[[TEMP_4:.*]] = mhlo.maximum %[[TEMP_1]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_5:.*]] = mhlo.constant dense<3.40282347E+38> : tensor<f32>
// CHECK: %[[TEMP_6:.*]] = mhlo.sqrt %[[TEMP_5]] : tensor<f32>
// CHECK: %[[TEMP_7:.*]] = mhlo.constant dense<8.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_8:.*]] = mhlo.divide %[[TEMP_6]], %[[TEMP_7]] : tensor<f32>
// CHECK: %[[TEMP_9:.*]] = mhlo.compare  GE, %[[TEMP_4]], %[[TEMP_8]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_10:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_11:.*]] = mhlo.compare  LE, %[[TEMP_1]], %[[TEMP_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_12:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK: %[[TEMP_13:.*]] = mhlo.add %[[TEMP_1]], %[[TEMP_10]] : tensor<f32>
// CHECK: %[[TEMP_14:.*]] = mhlo.abs %[[TEMP_13]] : tensor<f32>
// CHECK: %[[TEMP_15:.*]] = mhlo.maximum %[[TEMP_14]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_16:.*]] = mhlo.minimum %[[TEMP_14]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_17:.*]] = mhlo.compare  EQ, %[[TEMP_15]], %[[TEMP_16]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_18:.*]] = mhlo.constant dense<1.41421354> : tensor<f32>
// CHECK: %[[TEMP_19:.*]] = mhlo.multiply %[[TEMP_18]], %[[TEMP_15]] : tensor<f32>
// CHECK: %[[TEMP_20:.*]] = mhlo.divide %[[TEMP_16]], %[[TEMP_15]] : tensor<f32>
// CHECK: %[[TEMP_21:.*]] = mhlo.multiply %[[TEMP_20]], %[[TEMP_20]] : tensor<f32>
// CHECK: %[[TEMP_22:.*]] = mhlo.add %[[TEMP_10]], %[[TEMP_21]] : tensor<f32>
// CHECK: %[[TEMP_23:.*]] = mhlo.sqrt %[[TEMP_22]] : tensor<f32>
// CHECK: %[[TEMP_24:.*]] = mhlo.compare  EQ, %[[TEMP_23]], %[[TEMP_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_25:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_26:.*]] = mhlo.compare  GT, %[[TEMP_21]], %[[TEMP_25]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_27:.*]] = mhlo.and %[[TEMP_24]], %[[TEMP_26]] : tensor<i1>
// CHECK: %[[TEMP_28:.*]] = mhlo.multiply %[[TEMP_15]], %[[TEMP_21]] : tensor<f32>
// CHECK: %[[TEMP_29:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_30:.*]] = mhlo.divide %[[TEMP_28]], %[[TEMP_29]] : tensor<f32>
// CHECK: %[[TEMP_31:.*]] = mhlo.add %[[TEMP_15]], %[[TEMP_30]] : tensor<f32>
// CHECK: %[[TEMP_32:.*]] = mhlo.multiply %[[TEMP_15]], %[[TEMP_23]] : tensor<f32>
// CHECK: %[[TEMP_33:.*]] = mhlo.select %[[TEMP_27]], %[[TEMP_31]], %[[TEMP_32]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_34:.*]] = mhlo.select %[[TEMP_17]], %[[TEMP_19]], %[[TEMP_33]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_35:.*]] = mhlo.subtract %[[TEMP_1]], %[[TEMP_10]] : tensor<f32>
// CHECK: %[[TEMP_36:.*]] = mhlo.abs %[[TEMP_35]] : tensor<f32>
// CHECK: %[[TEMP_37:.*]] = mhlo.maximum %[[TEMP_36]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_38:.*]] = mhlo.minimum %[[TEMP_36]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_39:.*]] = mhlo.compare  EQ, %[[TEMP_37]], %[[TEMP_38]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_40:.*]] = mhlo.multiply %[[TEMP_18]], %[[TEMP_37]] : tensor<f32>
// CHECK: %[[TEMP_41:.*]] = mhlo.divide %[[TEMP_38]], %[[TEMP_37]] : tensor<f32>
// CHECK: %[[TEMP_42:.*]] = mhlo.multiply %[[TEMP_41]], %[[TEMP_41]] : tensor<f32>
// CHECK: %[[TEMP_43:.*]] = mhlo.add %[[TEMP_10]], %[[TEMP_42]] : tensor<f32>
// CHECK: %[[TEMP_44:.*]] = mhlo.sqrt %[[TEMP_43]] : tensor<f32>
// CHECK: %[[TEMP_45:.*]] = mhlo.compare  EQ, %[[TEMP_44]], %[[TEMP_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_46:.*]] = mhlo.compare  GT, %[[TEMP_42]], %[[TEMP_25]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_47:.*]] = mhlo.and %[[TEMP_45]], %[[TEMP_46]] : tensor<i1>
// CHECK: %[[TEMP_48:.*]] = mhlo.multiply %[[TEMP_37]], %[[TEMP_42]] : tensor<f32>
// CHECK: %[[TEMP_49:.*]] = mhlo.divide %[[TEMP_48]], %[[TEMP_29]] : tensor<f32>
// CHECK: %[[TEMP_50:.*]] = mhlo.add %[[TEMP_37]], %[[TEMP_49]] : tensor<f32>
// CHECK: %[[TEMP_51:.*]] = mhlo.multiply %[[TEMP_37]], %[[TEMP_44]] : tensor<f32>
// CHECK: %[[TEMP_52:.*]] = mhlo.select %[[TEMP_47]], %[[TEMP_50]], %[[TEMP_51]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_53:.*]] = mhlo.select %[[TEMP_39]], %[[TEMP_40]], %[[TEMP_52]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_54:.*]] = mhlo.add %[[TEMP_34]], %[[TEMP_53]] : tensor<f32>
// CHECK: %[[TEMP_55:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_54]] : tensor<f32>
// CHECK: %[[TEMP_56:.*]] = mhlo.add %[[TEMP_55]], %[[TEMP_1]] : tensor<f32>
// CHECK: %[[TEMP_57:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_56]] : tensor<f32>
// CHECK: %[[TEMP_58:.*]] = mhlo.multiply %[[TEMP_3]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_59:.*]] = mhlo.add %[[TEMP_34]], %[[TEMP_13]] : tensor<f32>
// CHECK: %[[TEMP_60:.*]] = mhlo.divide %[[TEMP_58]], %[[TEMP_59]] : tensor<f32>
// CHECK: %[[TEMP_61:.*]] = mhlo.subtract %[[TEMP_53]], %[[TEMP_35]] : tensor<f32>
// CHECK: %[[TEMP_62:.*]] = mhlo.add %[[TEMP_60]], %[[TEMP_61]] : tensor<f32>
// CHECK: %[[TEMP_63:.*]] = mhlo.multiply %[[TEMP_57]], %[[TEMP_62]] : tensor<f32>
// CHECK: %[[TEMP_64:.*]] = mhlo.sqrt %[[TEMP_63]] : tensor<f32>
// CHECK: %[[TEMP_65:.*]] = mhlo.divide %[[TEMP_57]], %[[TEMP_59]] : tensor<f32>
// CHECK: %[[TEMP_66:.*]] = mhlo.add %[[TEMP_53]], %[[TEMP_35]] : tensor<f32>
// CHECK: %[[TEMP_67:.*]] = mhlo.divide %[[TEMP_57]], %[[TEMP_66]] : tensor<f32>
// CHECK: %[[TEMP_68:.*]] = mhlo.add %[[TEMP_65]], %[[TEMP_67]] : tensor<f32>
// CHECK: %[[TEMP_69:.*]] = mhlo.sqrt %[[TEMP_68]] : tensor<f32>
// CHECK: %[[TEMP_70:.*]] = mhlo.multiply %[[TEMP_3]], %[[TEMP_69]] : tensor<f32>
// CHECK: %[[TEMP_71:.*]] = mhlo.select %[[TEMP_11]], %[[TEMP_64]], %[[TEMP_70]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_72:.*]] = mhlo.select %[[TEMP_9]], %[[TEMP_3]], %[[TEMP_71]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_73:.*]] = mhlo.constant dense<9.99999995E+11> : tensor<f32>
// CHECK: %[[TEMP_74:.*]] = mhlo.multiply %[[TEMP_8]], %[[TEMP_73]] : tensor<f32>
// CHECK: %[[TEMP_75:.*]] = mhlo.compare  LT, %[[TEMP_1]], %[[TEMP_74]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_76:.*]] = mhlo.constant dense<9.99999997E-7> : tensor<f32>
// CHECK: %[[TEMP_77:.*]] = mhlo.multiply %[[TEMP_8]], %[[TEMP_76]] : tensor<f32>
// CHECK: %[[TEMP_78:.*]] = mhlo.constant dense<1.000000e+02> : tensor<f32>
// CHECK: %[[TEMP_79:.*]] = mhlo.multiply %[[TEMP_8]], %[[TEMP_78]] : tensor<f32>
// CHECK: %[[TEMP_80:.*]] = mhlo.select %[[TEMP_75]], %[[TEMP_77]], %[[TEMP_79]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_81:.*]] = mhlo.compare  GE, %[[TEMP_3]], %[[TEMP_80]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_82:.*]] = mhlo.select %[[TEMP_81]], %[[TEMP_3]], %[[TEMP_1]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_83:.*]] = mhlo.select %[[TEMP_81]], %[[TEMP_80]], %[[TEMP_8]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_84:.*]] = mhlo.compare  GE, %[[TEMP_82]], %[[TEMP_83]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_85:.*]] = mhlo.log %[[TEMP_29]] : tensor<f32>
// CHECK: %[[TEMP_86:.*]] = mhlo.log %[[TEMP_82]] : tensor<f32>
// CHECK: %[[TEMP_87:.*]] = mhlo.add %[[TEMP_85]], %[[TEMP_86]] : tensor<f32>
// CHECK: %[[TEMP_88:.*]] = mhlo.constant dense<0x7F800000> : tensor<f32>
// CHECK: %[[TEMP_89:.*]] = mhlo.compare  EQ, %[[TEMP_3]], %[[TEMP_88]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_90:.*]] = mhlo.not %[[TEMP_89]] : tensor<i1>
// CHECK: %[[TEMP_91:.*]] = mhlo.and %[[TEMP_81]], %[[TEMP_90]] : tensor<i1>
// CHECK: %[[TEMP_92:.*]] = mhlo.divide %[[TEMP_1]], %[[TEMP_3]] : tensor<f32>
// CHECK: %[[TEMP_93:.*]] = mhlo.select %[[TEMP_91]], %[[TEMP_92]], %[[TEMP_25]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_94:.*]] = mhlo.multiply %[[TEMP_93]], %[[TEMP_93]] : tensor<f32>
// CHECK: %[[TEMP_95:.*]] = mhlo.log_plus_one %[[TEMP_94]] : tensor<f32>
// CHECK: %[[TEMP_96:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_95]] : tensor<f32>
// CHECK: %[[TEMP_97:.*]] = mhlo.add %[[TEMP_87]], %[[TEMP_96]] : tensor<f32>
// CHECK: %[[TEMP_98:.*]] = mhlo.constant dense<1.17549435E-38> : tensor<f32>
// CHECK: %[[TEMP_99:.*]] = mhlo.sqrt %[[TEMP_98]] : tensor<f32>
// CHECK: %[[TEMP_100:.*]] = mhlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_101:.*]] = mhlo.multiply %[[TEMP_99]], %[[TEMP_100]] : tensor<f32>
// CHECK: %[[TEMP_102:.*]] = mhlo.compare  LT, %[[TEMP_3]], %[[TEMP_101]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_103:.*]] = mhlo.compare  LT, %[[TEMP_1]], %[[TEMP_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_104:.*]] = mhlo.and %[[TEMP_102]], %[[TEMP_103]] : tensor<i1>
// CHECK: %[[TEMP_105:.*]] = mhlo.multiply %[[TEMP_13]], %[[TEMP_35]] : tensor<f32>
// CHECK: %[[TEMP_106:.*]] = mhlo.add %[[TEMP_55]], %[[TEMP_10]] : tensor<f32>
// CHECK: %[[TEMP_107:.*]] = mhlo.divide %[[TEMP_105]], %[[TEMP_106]] : tensor<f32>
// CHECK: %[[TEMP_108:.*]] = mhlo.negate %[[TEMP_107]] : tensor<f32>
// CHECK: %[[TEMP_109:.*]] = mhlo.compare  GE, %[[TEMP_1]], %[[TEMP_10]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_110:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_58]] : tensor<f32>
// CHECK: %[[TEMP_111:.*]] = mhlo.divide %[[TEMP_110]], %[[TEMP_59]] : tensor<f32>
// CHECK: %[[TEMP_112:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_66]] : tensor<f32>
// CHECK: %[[TEMP_113:.*]] = mhlo.add %[[TEMP_111]], %[[TEMP_112]] : tensor<f32>
// CHECK: %[[TEMP_114:.*]] = mhlo.constant dense<1.500000e+00> : tensor<f32>
// CHECK: %[[TEMP_115:.*]] = mhlo.compare  LE, %[[TEMP_55]], %[[TEMP_114]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_116:.*]] = mhlo.divide %[[TEMP_110]], %[[TEMP_61]] : tensor<f32>
// CHECK: %[[TEMP_117:.*]] = mhlo.add %[[TEMP_111]], %[[TEMP_116]] : tensor<f32>
// CHECK: %[[TEMP_118:.*]] = mhlo.subtract %[[TEMP_55]], %[[TEMP_10]] : tensor<f32>
// CHECK: %[[TEMP_119:.*]] = mhlo.select %[[TEMP_115]], %[[TEMP_117]], %[[TEMP_118]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_120:.*]] = mhlo.select %[[TEMP_109]], %[[TEMP_113]], %[[TEMP_119]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_121:.*]] = mhlo.select %[[TEMP_104]], %[[TEMP_108]], %[[TEMP_120]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_122:.*]] = mhlo.multiply %[[TEMP_121]], %[[TEMP_106]] : tensor<f32>
// CHECK: %[[TEMP_123:.*]] = mhlo.sqrt %[[TEMP_122]] : tensor<f32>
// CHECK: %[[TEMP_124:.*]] = mhlo.divide %[[TEMP_3]], %[[TEMP_123]] : tensor<f32>
// CHECK: %[[TEMP_125:.*]] = mhlo.add %[[TEMP_121]], %[[TEMP_123]] : tensor<f32>
// CHECK: %[[TEMP_126:.*]] = mhlo.log_plus_one %[[TEMP_125]] : tensor<f32>
// CHECK: %[[TEMP_127:.*]] = mhlo.select %[[TEMP_104]], %[[TEMP_124]], %[[TEMP_126]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_128:.*]] = mhlo.select %[[TEMP_84]], %[[TEMP_97]], %[[TEMP_127]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_129:.*]] = mhlo.complex %[[TEMP_72]], %[[TEMP_128]] : tensor<complex<f32>>
// CHECK: %[[TEMP_130:.*]] = mhlo.imag %[[TEMP_129]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_131:.*]] = mhlo.imag %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_132:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[TEMP_133:.*]] = mhlo.compare  LT, %[[TEMP_131]], %[[TEMP_132]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[TEMP_134:.*]] = mhlo.real %[[TEMP_129]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_135:.*]] = mhlo.real %[[TEMP_arg0:.*]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK: %[[TEMP_136:.*]] = mhlo.atan2 %[[TEMP_134]], %[[TEMP_135]] : tensor<f32>
// CHECK: %[[TEMP_137:.*]] = mhlo.negate %[[TEMP_136]] : tensor<f32>
// CHECK: %[[TEMP_138:.*]] = mhlo.select %[[TEMP_133]], %[[TEMP_137]], %[[TEMP_136]] : tensor<i1>, tensor<f32>
// CHECK: %[[TEMP_139:.*]] = mhlo.complex %[[TEMP_130]], %[[TEMP_138]] : tensor<complex<f32>>
// CHECK: return %[[TEMP_139]] : tensor<complex<f32>>
func.func @acosh_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "chlo.acosh"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// CHECK-LABEL: @erfc_f64
// CHECK-SAME: %[[ARG:.*]]: tensor<f64>
func.func @erfc_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK-NEXT: %[[TMP_0:.*]] = mhlo.multiply %[[ARG]], %[[ARG]]
  // CHECK-NEXT: %[[TMP_1:.*]] = mhlo.negate %[[TMP_0]]
  // CHECK-NEXT: %[[TMP_2:.*]] = mhlo.exponential %[[TMP_1]]
  // CHECK-NEXT: %[[TMP_3:.*]] = mhlo.abs %[[ARG]]
  // CHECK-NEXT: %[[TMP_6:.*]] = mhlo.constant dense<2.4619698147353052E-10>
  // CHECK-NEXT: %[[TMP_8:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_3]]
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
  // CHECK-NEXT: %[[TMP_35:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_37:.*]] = mhlo.multiply %[[TMP_35]], %[[TMP_3]]
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
  // CHECK-NEXT: %[[TMP_64:.*]] = mhlo.constant dense<0.56418958354775506>
  // CHECK-NEXT: %[[TMP_66:.*]] = mhlo.multiply %[[TMP_64]], %[[TMP_3]]
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
  // CHECK-NEXT: %[[TMP_84:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_86:.*]] = mhlo.multiply %[[TMP_84]], %[[TMP_3]]
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
  // CHECK-NEXT: %[[TMP_106:.*]] = mhlo.compare LT, %[[TMP_3]], %[[TMP_105]]
  // CHECK-NEXT: %[[TMP_107:.*]] = mhlo.select %[[TMP_106]], %[[TMP_61]], %[[TMP_104]]
  // CHECK-NEXT: %[[TMP_108:.*]] = mhlo.constant dense<-709.78271289338397>
  // CHECK-NEXT: %[[TMP_109:.*]] = mhlo.compare LT, %[[TMP_1]], %[[TMP_108]]
  // CHECK-NEXT: %[[TMP_110:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[TMP_111:.*]] = mhlo.select %[[TMP_109]], %[[TMP_110]], %[[TMP_107]]
  // CHECK-NEXT: %[[TMP_113:.*]] = mhlo.compare LT, %[[ARG]], %[[TMP_110]]
  // CHECK-NEXT: %[[TMP_114:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK-NEXT: %[[TMP_115:.*]] = mhlo.subtract %[[TMP_114]], %[[TMP_111]]
  // CHECK-NEXT: %[[TMP_116:.*]] = mhlo.select %[[TMP_113]], %[[TMP_115]], %[[TMP_111]]
  // CHECK-NEXT: %[[TMP_117:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_118:.*]] = mhlo.multiply %[[ARG]], %[[ARG]]
  // CHECK-NEXT: %[[TMP_121:.*]] = mhlo.constant dense<9.6049737398705161>
  // CHECK-NEXT: %[[TMP_123:.*]] = mhlo.multiply %[[TMP_121]], %[[TMP_118]]
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
  // CHECK-NEXT: %[[TMP_138:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_140:.*]] = mhlo.multiply %[[TMP_138]], %[[TMP_118]]
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
  // CHECK-NEXT: %[[TMP_157:.*]] = mhlo.abs %[[ARG]]
  // CHECK-NEXT: %[[TMP_159:.*]] = mhlo.compare LT, %[[TMP_157]], %[[TMP_117]]
  // CHECK-NEXT: %[[RESULT:.*]] = mhlo.select %[[TMP_159]], %[[TMP_156]], %[[TMP_116]]
  // CHECK-NEXT: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @erfc_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @erfc_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_0:.*]] = mhlo.multiply %[[ARG]], %[[ARG]]
  // CHECK: %[[TMP_1:.*]] = mhlo.negate %[[TMP_0]]
  // CHECK: %[[TMP_2:.*]] = mhlo.abs %[[ARG]]
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_4:.*]] = mhlo.divide %[[TMP_3]], %[[TMP_0]]
  // CHECK: %[[TMP_5:.*]] = mhlo.exponential %[[TMP_1]]
  // CHECK: %[[TMP_7:.*]] = mhlo.divide %[[TMP_3]], %[[TMP_2]]
  // CHECK: %[[TMP_8:.*]] = mhlo.multiply %[[TMP_5]], %[[TMP_7]]
  // CHECK: %[[TMP_9:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_10:.*]] = mhlo.compare LT, %[[TMP_2]], %[[TMP_9]]
  // CHECK: %[[TMP_13:.*]] = mhlo.constant dense<2.326820e-02>
  // CHECK: %[[TMP_15:.*]] = mhlo.multiply %[[TMP_13]], %[[TMP_4]]
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
  // CHECK: %[[TMP_41:.*]] = mhlo.constant dense<-10.477664>
  // CHECK: %[[TMP_43:.*]] = mhlo.multiply %[[TMP_41]], %[[TMP_4]]
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
  // CHECK: %[[TMP_64:.*]] = mhlo.select %[[TMP_10]], %[[TMP_38]], %[[TMP_63]]
  // CHECK: %[[TMP_65:.*]] = mhlo.multiply %[[TMP_8]], %[[TMP_64]]
  // CHECK: %[[TMP_66:.*]] = mhlo.constant dense<-88.7228394>
  // CHECK: %[[TMP_67:.*]] = mhlo.compare LT, %[[TMP_1]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_69:.*]] = mhlo.select %[[TMP_67]], %[[TMP_68]], %[[TMP_65]]
  // CHECK: %[[TMP_71:.*]] = mhlo.compare LT, %[[ARG]], %[[TMP_68]]
  // CHECK: %[[TMP_73:.*]] = mhlo.subtract %[[TMP_9]], %[[TMP_69]]
  // CHECK: %[[TMP_74:.*]] = mhlo.select %[[TMP_71]], %[[TMP_73]], %[[TMP_69]]
  // CHECK: %[[TMP_75:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_76:.*]] = mhlo.multiply %[[ARG]], %[[ARG]]
  // CHECK: %[[TMP_79:.*]] = mhlo.constant dense<7.85386146E-5>
  // CHECK: %[[TMP_81:.*]] = mhlo.multiply %[[TMP_79]], %[[TMP_76]]
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
  // CHECK: %[[TMP_101:.*]] = mhlo.abs %[[ARG]]
  // CHECK: %[[TMP_103:.*]] = mhlo.compare LT, %[[TMP_101]], %[[TMP_75]]
  // CHECK: %[[RESULT:.*]] = mhlo.select %[[TMP_103]], %[[TMP_100]], %[[TMP_74]]
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @erfc_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func.func @erfc_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: mhlo.convert %[[ARG]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = mhlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @erfc_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func.func @erfc_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  // CHECK: mhlo.convert %[[ARG]] : (tensor<bf16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = mhlo.convert %{{.*}} : (tensor<f32>) -> tensor<bf16>
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erfc"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %1 : tensor<bf16>
}

// -----

// CHECK-LABEL: @is_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @is_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  // CHECK: %[[ABS:.*]] = mhlo.abs %arg0 : tensor<f32>
  // CHECK: %[[POS_INF:.*]] = mhlo.constant dense<0x7F800000> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = mhlo.compare EQ, %[[ABS]], %[[POS_INF]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: return %[[RESULT]] : tensor<i1>
  %1 = chlo.is_inf %arg : tensor<f32> -> tensor<i1>
  func.return %1 : tensor<i1>
}

// -----

// CHECK-LABEL: @is_pos_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @is_pos_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  // CHECK: %[[POS_INF:.*]] = mhlo.constant dense<0x7F800000> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = mhlo.compare EQ, %[[ARG]], %[[POS_INF]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: return %[[RESULT]] : tensor<i1>
  %1 = chlo.is_pos_inf %arg : tensor<f32> -> tensor<i1>
  func.return %1 : tensor<i1>
}

// -----

// CHECK-LABEL: @is_neg_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @is_neg_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  // CHECK: %[[NEG_INF:.*]] = mhlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = mhlo.compare EQ, %[[ARG]], %[[NEG_INF]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: return %[[RESULT]] : tensor<i1>
  %1 = chlo.is_neg_inf %arg : tensor<f32> -> tensor<i1>
  func.return %1 : tensor<i1>
}

// -----

// CHECK-LABEL: @lgamma_f64
// CHECK-SAME: (%[[ARG:.*]]: tensor<f64>)
func.func @lgamma_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK: %[[TMP_1:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_9:.*]] = mhlo.compare LT, %[[ARG]], %[[TMP_1]]
  // CHECK: %[[TMP_10:.*]] = mhlo.negate %[[ARG]]
  // CHECK: %[[TMP_2:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_11:.*]] = mhlo.subtract %[[ARG]], %[[TMP_2]]
  // CHECK: %[[TMP_12:.*]] = mhlo.select %[[TMP_9]], %[[TMP_10]], %[[TMP_11]]
  // CHECK-DAG: %[[TMP_8:.*]] = mhlo.constant dense<0.99999999999980993>
  // CHECK-DAG: %[[TMP_13:.*]] = mhlo.constant dense<676.5203681218851>
  // CHECK-DAG: %[[TMP_14:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_12]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = mhlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = mhlo.add %[[TMP_8]], %[[TMP_16]]
  // CHECK-DAG: %[[TMP_18:.*]] = mhlo.constant dense<-1259.1392167224028>
  // CHECK-DAG: %[[TMP_19:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_12]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK-DAG: %[[TMP_23:.*]] = mhlo.constant dense<771.32342877765313>
  // CHECK-DAG: %[[TMP_24:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = mhlo.add %[[TMP_12]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = mhlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK-DAG: %[[TMP_28:.*]] = mhlo.constant dense<-176.61502916214059>
  // CHECK-DAG: %[[TMP_29:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = mhlo.add %[[TMP_12]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = mhlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK-DAG: %[[TMP_33:.*]] = mhlo.constant dense<12.507343278686905>
  // CHECK-DAG: %[[TMP_34:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = mhlo.add %[[TMP_12]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK-DAG: %[[TMP_38:.*]] = mhlo.constant dense<-0.13857109526572012>
  // CHECK-DAG: %[[TMP_39:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = mhlo.add %[[TMP_12]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = mhlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK-DAG: %[[TMP_43:.*]] = mhlo.constant dense<9.9843695780195716E-6>
  // CHECK-DAG: %[[TMP_44:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = mhlo.add %[[TMP_12]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = mhlo.constant dense<1.5056327351493116E-7>
  // CHECK-DAG: %[[TMP_49:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_12]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_53:.*]] = mhlo.add %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_7:.*]] = mhlo.constant dense<2.0149030205422647>
  // CHECK: %[[TMP_54:.*]] = mhlo.divide %[[TMP_12]], %[[TMP_6]]
  // CHECK: %[[TMP_55:.*]] = mhlo.log_plus_one %[[TMP_54]]
  // CHECK: %[[TMP_56:.*]] = mhlo.add %[[TMP_7]], %[[TMP_55]]
  // CHECK: %[[TMP_57:.*]] = mhlo.divide %[[TMP_53]], %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_12]], %[[TMP_1]]
  // CHECK: %[[TMP_59:.*]] = mhlo.subtract %[[TMP_58]], %[[TMP_57]]
  // CHECK: %[[TMP_60:.*]] = mhlo.multiply %[[TMP_59]], %[[TMP_56]]
  // CHECK: %[[TMP_61:.*]] = mhlo.log %[[TMP_52]]
  // CHECK: %[[TMP_5:.*]] = mhlo.constant dense<0.91893853320467266>
  // CHECK: %[[TMP_62:.*]] = mhlo.add %[[TMP_5]], %[[TMP_60]]
  // CHECK: %[[TMP_63:.*]] = mhlo.add %[[TMP_62]], %[[TMP_61]]
  // CHECK: %[[TMP_64:.*]] = mhlo.abs %[[ARG]]
  // CHECK: %[[TMP_65:.*]] = mhlo.floor %[[TMP_64]]
  // CHECK: %[[TMP_66:.*]] = mhlo.subtract %[[TMP_64]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = mhlo.compare LT, %[[TMP_1]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = mhlo.subtract %[[TMP_2]], %[[TMP_66]]
  // CHECK: %[[TMP_69:.*]] = mhlo.select %[[TMP_67]], %[[TMP_68]], %[[TMP_66]]
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_70:.*]] = mhlo.multiply %[[TMP_3]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = mhlo.sine %[[TMP_70]]
  // CHECK: %[[TMP_72:.*]] = mhlo.log %[[TMP_71]]
  // CHECK: %[[TMP_4:.*]] = mhlo.constant dense<1.1447298858494002>
  // CHECK: %[[TMP_75:.*]] = mhlo.subtract %[[TMP_4]], %[[TMP_72]]
  // CHECK: %[[TMP_76:.*]] = mhlo.subtract %[[TMP_75]], %[[TMP_63]]
  // CHECK: %[[TMP_73:.*]] = mhlo.is_finite %[[TMP_72]]
  // CHECK: %[[TMP_74:.*]] = mhlo.negate %[[TMP_72]]
  // CHECK: %[[TMP_77:.*]] = mhlo.select %[[TMP_73]], %[[TMP_76]], %[[TMP_74]]
  // CHECK: %[[TMP_78:.*]] = mhlo.select %[[TMP_9]], %[[TMP_77]], %[[TMP_63]]
  // CHECK: %[[TMP_79:.*]] = mhlo.abs %[[ARG]]
  // CHECK: %[[TMP_80:.*]] = mhlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_81:.*]] = mhlo.compare EQ, %[[TMP_79]], %[[TMP_80]]
  // CHECK: %[[TMP_0:.*]] = mhlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_82:.*]] = mhlo.select %[[TMP_81]], %[[TMP_0]], %[[TMP_78]]
  // CHECK: return %[[TMP_82]]
  %1 = chlo.lgamma %arg : tensor<f64> -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @lgamma_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @lgamma_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_1:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_9:.*]] = mhlo.compare LT, %[[ARG]], %[[TMP_1]]
  // CHECK: %[[TMP_10:.*]] = mhlo.negate %[[ARG]]
  // CHECK: %[[TMP_2:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_11:.*]] = mhlo.subtract %[[ARG]], %[[TMP_2]]
  // CHECK: %[[TMP_12:.*]] = mhlo.select %[[TMP_9]], %[[TMP_10]], %[[TMP_11]]
  // CHECK-DAG: %[[TMP_8:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_13:.*]] = mhlo.constant dense<676.520386>
  // CHECK-DAG: %[[TMP_14:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_12]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = mhlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = mhlo.add %[[TMP_8]], %[[TMP_16]]
  // CHECK-DAG: %[[TMP_18:.*]] = mhlo.constant dense<-1259.13916>
  // CHECK-DAG: %[[TMP_19:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_12]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK-DAG: %[[TMP_23:.*]] = mhlo.constant dense<771.323425>
  // CHECK-DAG: %[[TMP_24:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = mhlo.add %[[TMP_12]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = mhlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK-DAG: %[[TMP_28:.*]] = mhlo.constant dense<-176.615036>
  // CHECK-DAG: %[[TMP_29:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = mhlo.add %[[TMP_12]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = mhlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK-DAG: %[[TMP_33:.*]] = mhlo.constant dense<12.5073433>
  // CHECK-DAG: %[[TMP_34:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = mhlo.add %[[TMP_12]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK-DAG: %[[TMP_38:.*]] = mhlo.constant dense<-0.138571098>
  // CHECK-DAG: %[[TMP_39:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = mhlo.add %[[TMP_12]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = mhlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK-DAG: %[[TMP_43:.*]] = mhlo.constant dense<9.98436917E-6>
  // CHECK-DAG: %[[TMP_44:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = mhlo.add %[[TMP_12]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = mhlo.constant dense<1.50563267E-7>
  // CHECK-DAG: %[[TMP_49:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_12]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_53:.*]] = mhlo.add %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_7:.*]] = mhlo.constant dense<2.01490307>
  // CHECK: %[[TMP_54:.*]] = mhlo.divide %[[TMP_12]], %[[TMP_6]]
  // CHECK: %[[TMP_55:.*]] = mhlo.log_plus_one %[[TMP_54]]
  // CHECK: %[[TMP_56:.*]] = mhlo.add %[[TMP_7]], %[[TMP_55]]
  // CHECK: %[[TMP_57:.*]] = mhlo.divide %[[TMP_53]], %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_12]], %[[TMP_1]]
  // CHECK: %[[TMP_59:.*]] = mhlo.subtract %[[TMP_58]], %[[TMP_57]]
  // CHECK: %[[TMP_60:.*]] = mhlo.multiply %[[TMP_59]], %[[TMP_56]]
  // CHECK: %[[TMP_61:.*]] = mhlo.log %[[TMP_52]]
  // CHECK: %[[TMP_5:.*]] = mhlo.constant dense<0.918938517>
  // CHECK: %[[TMP_62:.*]] = mhlo.add %[[TMP_5]], %[[TMP_60]]
  // CHECK: %[[TMP_63:.*]] = mhlo.add %[[TMP_62]], %[[TMP_61]]
  // CHECK: %[[TMP_64:.*]] = mhlo.abs %[[ARG]]
  // CHECK: %[[TMP_65:.*]] = mhlo.floor %[[TMP_64]]
  // CHECK: %[[TMP_66:.*]] = mhlo.subtract %[[TMP_64]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = mhlo.compare LT, %[[TMP_1]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = mhlo.subtract %[[TMP_2]], %[[TMP_66]]
  // CHECK: %[[TMP_69:.*]] = mhlo.select %[[TMP_67]], %[[TMP_68]], %[[TMP_66]]
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<3.14159274>
  // CHECK: %[[TMP_70:.*]] = mhlo.multiply %[[TMP_3]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = mhlo.sine %[[TMP_70]]
  // CHECK: %[[TMP_72:.*]] = mhlo.log %[[TMP_71]]
  // CHECK: %[[TMP_4:.*]] = mhlo.constant dense<1.14472985>
  // CHECK: %[[TMP_75:.*]] = mhlo.subtract %[[TMP_4]], %[[TMP_72]]
  // CHECK: %[[TMP_76:.*]] = mhlo.subtract %[[TMP_75]], %[[TMP_63]]
  // CHECK: %[[TMP_73:.*]] = mhlo.is_finite %[[TMP_72]]
  // CHECK: %[[TMP_74:.*]] = mhlo.negate %[[TMP_72]]
  // CHECK: %[[TMP_77:.*]] = mhlo.select %[[TMP_73]], %[[TMP_76]], %[[TMP_74]]
  // CHECK: %[[TMP_78:.*]] = mhlo.select %[[TMP_9]], %[[TMP_77]], %[[TMP_63]]
  // CHECK: %[[TMP_79:.*]] = mhlo.abs %[[ARG]]
  // CHECK: %[[TMP_80:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_81:.*]] = mhlo.compare EQ, %[[TMP_79]], %[[TMP_80]]
  // CHECK: %[[TMP_0:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_82:.*]] = mhlo.select %[[TMP_81]], %[[TMP_0]], %[[TMP_78]]
  // CHECK: return %[[TMP_82]]
  %1 = chlo.lgamma %arg : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @lgamma_f16
// CHECK-SAME: (%[[ARG:.*]]: tensor<f16>)
func.func @lgamma_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: mhlo.convert %[[ARG]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = mhlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.lgamma %arg : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @digamma_f64
// CHECK-SAME: (%[[ARG:.*]]: tensor<f64>)
func.func @digamma_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK: %[[TMP_0:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_1:.*]] = mhlo.compare LT, %arg0, %[[TMP_0]]
  // CHECK: %[[TMP_2:.*]] = mhlo.negate %arg0
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_4:.*]] = mhlo.subtract %arg0, %[[TMP_3]]
  // CHECK: %[[TMP_5:.*]] = mhlo.select %[[TMP_1]], %[[TMP_2]], %[[TMP_4]]
  // CHECK-DAG: %[[TMP_6:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_7:.*]] = mhlo.constant dense<0.99999999999980993>
  // CHECK-DAG: %[[TMP_8:.*]] = mhlo.constant dense<676.5203681218851>
  // CHECK-DAG: %[[TMP_9:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = mhlo.add %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = mhlo.multiply %[[TMP_10]], %[[TMP_10]]
  // CHECK: %[[TMP_12:.*]] = mhlo.divide %[[TMP_8]], %[[TMP_11]]
  // CHECK: %[[TMP_13:.*]] = mhlo.subtract %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_14:.*]] = mhlo.divide %[[TMP_8]], %[[TMP_10]]
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_7]], %[[TMP_14]]
  // CHECK-DAG: %[[TMP_16:.*]] = mhlo.constant dense<-1259.1392167224028>
  // CHECK-DAG: %[[TMP_17:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_18:.*]] = mhlo.add %[[TMP_5]], %[[TMP_17]]
  // CHECK: %[[TMP_19:.*]] = mhlo.multiply %[[TMP_18]], %[[TMP_18]]
  // CHECK: %[[TMP_20:.*]] = mhlo.divide %[[TMP_16]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.subtract %[[TMP_13]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.divide %[[TMP_16]], %[[TMP_18]]
  // CHECK: %[[TMP_23:.*]] = mhlo.add %[[TMP_15]], %[[TMP_22]]
  // CHECK-DAG: %[[TMP_24:.*]] = mhlo.constant dense<771.32342877765313>
  // CHECK-DAG: %[[TMP_25:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_26:.*]] = mhlo.add %[[TMP_5]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.multiply %[[TMP_26]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = mhlo.divide %[[TMP_24]], %[[TMP_27]]
  // CHECK: %[[TMP_29:.*]] = mhlo.subtract %[[TMP_21]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = mhlo.divide %[[TMP_24]], %[[TMP_26]]
  // CHECK: %[[TMP_31:.*]] = mhlo.add %[[TMP_23]], %[[TMP_30]]
  // CHECK-DAG: %[[TMP_32:.*]] = mhlo.constant dense<-176.61502916214059>
  // CHECK-DAG: %[[TMP_33:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_34:.*]] = mhlo.add %[[TMP_5]], %[[TMP_33]]
  // CHECK: %[[TMP_35:.*]] = mhlo.multiply %[[TMP_34]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_32]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.subtract %[[TMP_29]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = mhlo.divide %[[TMP_32]], %[[TMP_34]]
  // CHECK: %[[TMP_39:.*]] = mhlo.add %[[TMP_31]], %[[TMP_38]]
  // CHECK-DAG: %[[TMP_40:.*]] = mhlo.constant dense<12.507343278686905>
  // CHECK-DAG: %[[TMP_41:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_5]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = mhlo.multiply %[[TMP_42]], %[[TMP_42]]
  // CHECK: %[[TMP_44:.*]] = mhlo.divide %[[TMP_40]], %[[TMP_43]]
  // CHECK: %[[TMP_45:.*]] = mhlo.subtract %[[TMP_37]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_40]], %[[TMP_42]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_39]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = mhlo.constant dense<-0.13857109526572012>
  // CHECK-DAG: %[[TMP_49:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_5]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.multiply %[[TMP_50]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = mhlo.subtract %[[TMP_45]], %[[TMP_52]]
  // CHECK: %[[TMP_54:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_55:.*]] = mhlo.add %[[TMP_47]], %[[TMP_54]]
  // CHECK-DAG: %[[TMP_56:.*]] = mhlo.constant dense<9.9843695780195716E-6>
  // CHECK-DAG: %[[TMP_57:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_5]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = mhlo.multiply %[[TMP_58]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = mhlo.divide %[[TMP_56]], %[[TMP_59]]
  // CHECK: %[[TMP_61:.*]] = mhlo.subtract %[[TMP_53]], %[[TMP_60]]
  // CHECK: %[[TMP_62:.*]] = mhlo.divide %[[TMP_56]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = mhlo.add %[[TMP_55]], %[[TMP_62]]
  // CHECK-DAG: %[[TMP_64:.*]] = mhlo.constant dense<1.5056327351493116E-7>
  // CHECK-DAG: %[[TMP_65:.*]] = mhlo.constant dense<8.000000e+00>
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
  // CHECK: %[[TMP_76:.*]] = mhlo.log_plus_one %[[TMP_75]]
  // CHECK: %[[TMP_77:.*]] = mhlo.add %[[TMP_74]], %[[TMP_76]]
  // CHECK: %[[TMP_78:.*]] = mhlo.divide %[[TMP_69]], %[[TMP_71]]
  // CHECK: %[[TMP_79:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_80:.*]] = mhlo.divide %[[TMP_79]], %[[TMP_73]]
  // CHECK: %[[TMP_81:.*]] = mhlo.add %[[TMP_77]], %[[TMP_78]]
  // CHECK: %[[TMP_82:.*]] = mhlo.subtract %[[TMP_81]], %[[TMP_80]]
  // CHECK: %[[TMP_83:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_84:.*]] = mhlo.add %arg0, %[[TMP_83]]
  // CHECK: %[[TMP_85:.*]] = mhlo.floor %[[TMP_84]]
  // CHECK: %[[TMP_86:.*]] = mhlo.abs %[[TMP_85]]
  // CHECK: %[[TMP_87:.*]] = mhlo.add %arg0, %[[TMP_86]]
  // CHECK: %[[TMP_88:.*]] = mhlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_89:.*]] = mhlo.multiply %[[TMP_88]], %[[TMP_87]]
  // CHECK: %[[TMP_90:.*]] = mhlo.cosine %[[TMP_89]]
  // CHECK: %[[TMP_92:.*]] = mhlo.sine %[[TMP_89]]
  // CHECK: %[[TMP_91:.*]] = mhlo.multiply %[[TMP_88]], %[[TMP_90]]
  // CHECK: %[[TMP_93:.*]] = mhlo.divide %[[TMP_91]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = mhlo.subtract %[[TMP_82]], %[[TMP_93]]
  // CHECK: %[[TMP_95:.*]] = mhlo.select %[[TMP_1]], %[[TMP_94]], %[[TMP_82]]
  // CHECK: %[[TMP_96:.*]] = mhlo.compare LE, %arg0, %[[TMP_6]]
  // CHECK: %[[TMP_97:.*]] = mhlo.floor %arg0
  // CHECK: %[[TMP_98:.*]] = mhlo.compare EQ, %arg0, %[[TMP_97]]
  // CHECK: %[[TMP_99:.*]] = mhlo.and %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[RES:.*]] = mhlo.select %[[TMP_99]], %[[TMP_100]], %[[TMP_95]]
  // CHECK: return %[[RES]]
  %1 = chlo.digamma %arg : tensor<f64> -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @digamma_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @digamma_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[TMP_0:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_1:.*]] = mhlo.compare LT, %arg0, %[[TMP_0]]
  // CHECK: %[[TMP_2:.*]] = mhlo.negate %arg0
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_4:.*]] = mhlo.subtract %arg0, %[[TMP_3]]
  // CHECK: %[[TMP_5:.*]] = mhlo.select %[[TMP_1]], %[[TMP_2]], %[[TMP_4]]
  // CHECK-DAG: %[[TMP_6:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_7:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_8:.*]] = mhlo.constant dense<676.520386>
  // CHECK-DAG: %[[TMP_9:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = mhlo.add %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = mhlo.multiply %[[TMP_10]], %[[TMP_10]]
  // CHECK: %[[TMP_12:.*]] = mhlo.divide %[[TMP_8]], %[[TMP_11]]
  // CHECK: %[[TMP_13:.*]] = mhlo.subtract %[[TMP_6]], %[[TMP_12]]
  // CHECK: %[[TMP_14:.*]] = mhlo.divide %[[TMP_8]], %[[TMP_10]]
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_7]], %[[TMP_14]]
  // CHECK-DAG: %[[TMP_16:.*]] = mhlo.constant dense<-1259.13916>
  // CHECK-DAG: %[[TMP_17:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_18:.*]] = mhlo.add %[[TMP_5]], %[[TMP_17]]
  // CHECK: %[[TMP_19:.*]] = mhlo.multiply %[[TMP_18]], %[[TMP_18]]
  // CHECK: %[[TMP_20:.*]] = mhlo.divide %[[TMP_16]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.subtract %[[TMP_13]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.divide %[[TMP_16]], %[[TMP_18]]
  // CHECK: %[[TMP_23:.*]] = mhlo.add %[[TMP_15]], %[[TMP_22]]
  // CHECK-DAG: %[[TMP_24:.*]] = mhlo.constant dense<771.323425>
  // CHECK-DAG: %[[TMP_25:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_26:.*]] = mhlo.add %[[TMP_5]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.multiply %[[TMP_26]], %[[TMP_26]]
  // CHECK: %[[TMP_28:.*]] = mhlo.divide %[[TMP_24]], %[[TMP_27]]
  // CHECK: %[[TMP_29:.*]] = mhlo.subtract %[[TMP_21]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = mhlo.divide %[[TMP_24]], %[[TMP_26]]
  // CHECK: %[[TMP_31:.*]] = mhlo.add %[[TMP_23]], %[[TMP_30]]
  // CHECK-DAG: %[[TMP_32:.*]] = mhlo.constant dense<-176.615036>
  // CHECK-DAG: %[[TMP_33:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_34:.*]] = mhlo.add %[[TMP_5]], %[[TMP_33]]
  // CHECK: %[[TMP_35:.*]] = mhlo.multiply %[[TMP_34]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_32]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.subtract %[[TMP_29]], %[[TMP_36]]
  // CHECK: %[[TMP_38:.*]] = mhlo.divide %[[TMP_32]], %[[TMP_34]]
  // CHECK: %[[TMP_39:.*]] = mhlo.add %[[TMP_31]], %[[TMP_38]]
  // CHECK-DAG: %[[TMP_40:.*]] = mhlo.constant dense<12.5073433>
  // CHECK-DAG: %[[TMP_41:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_5]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = mhlo.multiply %[[TMP_42]], %[[TMP_42]]
  // CHECK: %[[TMP_44:.*]] = mhlo.divide %[[TMP_40]], %[[TMP_43]]
  // CHECK: %[[TMP_45:.*]] = mhlo.subtract %[[TMP_37]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_40]], %[[TMP_42]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_39]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = mhlo.constant dense<-0.138571098>
  // CHECK-DAG: %[[TMP_49:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_5]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.multiply %[[TMP_50]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = mhlo.subtract %[[TMP_45]], %[[TMP_52]]
  // CHECK: %[[TMP_54:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_55:.*]] = mhlo.add %[[TMP_47]], %[[TMP_54]]
  // CHECK-DAG: %[[TMP_56:.*]] = mhlo.constant dense<9.98436917E-6>
  // CHECK-DAG: %[[TMP_57:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_5]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = mhlo.multiply %[[TMP_58]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = mhlo.divide %[[TMP_56]], %[[TMP_59]]
  // CHECK: %[[TMP_61:.*]] = mhlo.subtract %[[TMP_53]], %[[TMP_60]]
  // CHECK: %[[TMP_62:.*]] = mhlo.divide %[[TMP_56]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = mhlo.add %[[TMP_55]], %[[TMP_62]]
  // CHECK-DAG: %[[TMP_64:.*]] = mhlo.constant dense<1.50563267E-7>
  // CHECK-DAG: %[[TMP_65:.*]] = mhlo.constant dense<8.000000e+00>
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
  // CHECK: %[[TMP_76:.*]] = mhlo.log_plus_one %[[TMP_75]]
  // CHECK: %[[TMP_77:.*]] = mhlo.add %[[TMP_74]], %[[TMP_76]]
  // CHECK: %[[TMP_78:.*]] = mhlo.divide %[[TMP_69]], %[[TMP_71]]
  // CHECK: %[[TMP_79:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_80:.*]] = mhlo.divide %[[TMP_79]], %[[TMP_73]]
  // CHECK: %[[TMP_81:.*]] = mhlo.add %[[TMP_77]], %[[TMP_78]]
  // CHECK: %[[TMP_82:.*]] = mhlo.subtract %[[TMP_81]], %[[TMP_80]]
  // CHECK: %[[TMP_83:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_84:.*]] = mhlo.add %arg0, %[[TMP_83]]
  // CHECK: %[[TMP_85:.*]] = mhlo.floor %[[TMP_84]]
  // CHECK: %[[TMP_86:.*]] = mhlo.abs %[[TMP_85]]
  // CHECK: %[[TMP_87:.*]] = mhlo.add %arg0, %[[TMP_86]]
  // CHECK: %[[TMP_88:.*]] = mhlo.constant dense<3.14159274>
  // CHECK: %[[TMP_89:.*]] = mhlo.multiply %[[TMP_88]], %[[TMP_87]]
  // CHECK: %[[TMP_90:.*]] = mhlo.cosine %[[TMP_89]]
  // CHECK: %[[TMP_92:.*]] = mhlo.sine %[[TMP_89]]
  // CHECK: %[[TMP_91:.*]] = mhlo.multiply %[[TMP_88]], %[[TMP_90]]
  // CHECK: %[[TMP_93:.*]] = mhlo.divide %[[TMP_91]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = mhlo.subtract %[[TMP_82]], %[[TMP_93]]
  // CHECK: %[[TMP_95:.*]] = mhlo.select %[[TMP_1]], %[[TMP_94]], %[[TMP_82]]
  // CHECK: %[[TMP_96:.*]] = mhlo.compare LE, %arg0, %[[TMP_6]]
  // CHECK: %[[TMP_97:.*]] = mhlo.floor %arg0
  // CHECK: %[[TMP_98:.*]] = mhlo.compare EQ, %arg0, %[[TMP_97]]
  // CHECK: %[[TMP_99:.*]] = mhlo.and %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.constant dense<0x7FC00000>
  // CHECK: %[[RES:.*]] = mhlo.select %[[TMP_99]], %[[TMP_100]], %[[TMP_95]]
  // CHECK: return %[[RES]]
  %1 = chlo.digamma %arg : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @digamma_f16
// CHECK-SAME: (%[[ARG:.*]]: tensor<f16>)
func.func @digamma_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: mhlo.convert %[[ARG]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = mhlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.digamma %arg : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @zeta_f16
// CHECK-SAME:  (%[[X:.*]]: tensor<f16>, %[[Q:.*]]: tensor<f16>) -> tensor<f16>
func.func @zeta_f16(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
  // CHECK: %[[TMP_0:.*]] = mhlo.convert %[[X]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[TMP_1:.*]] = mhlo.convert %[[Q]] : (tensor<f16>) -> tensor<f32>
  // CHECK-DAG: %[[TMP_2:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_3:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_4:.*]] = mhlo.negate %[[TMP_0]]
  // CHECK: %[[TMP_5:.*]] = mhlo.power %[[TMP_1]], %[[TMP_4]]
  // CHECK: %[[TMP_6:.*]] = mhlo.add %[[TMP_1]], %[[TMP_3]]
  // CHECK: %[[TMP_7:.*]] = mhlo.power %[[TMP_6]], %[[TMP_4]]
  // CHECK: %[[TMP_8:.*]] = mhlo.add %[[TMP_5]], %[[TMP_7]]
  // CHECK: %[[TMP_9:.*]] = mhlo.add %[[TMP_6]], %[[TMP_3]]
  // CHECK: %[[TMP_10:.*]] = mhlo.power %[[TMP_9]], %[[TMP_4]]
  // CHECK: %[[TMP_11:.*]] = mhlo.add %[[TMP_8]], %[[TMP_10]]
  // CHECK: %[[TMP_12:.*]] = mhlo.add %[[TMP_9]], %[[TMP_3]]
  // CHECK: %[[TMP_13:.*]] = mhlo.power %[[TMP_12]], %[[TMP_4]]
  // CHECK: %[[TMP_14:.*]] = mhlo.add %[[TMP_11]], %[[TMP_13]]
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_12]], %[[TMP_3]]
  // CHECK: %[[TMP_16:.*]] = mhlo.power %[[TMP_15]], %[[TMP_4]]
  // CHECK: %[[TMP_17:.*]] = mhlo.add %[[TMP_14]], %[[TMP_16]]
  // CHECK: %[[TMP_18:.*]] = mhlo.add %[[TMP_15]], %[[TMP_3]]
  // CHECK: %[[TMP_19:.*]] = mhlo.power %[[TMP_18]], %[[TMP_4]]
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_17]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.add %[[TMP_18]], %[[TMP_3]]
  // CHECK: %[[TMP_22:.*]] = mhlo.power %[[TMP_21]], %[[TMP_4]]
  // CHECK: %[[TMP_23:.*]] = mhlo.add %[[TMP_20]], %[[TMP_22]]
  // CHECK: %[[TMP_24:.*]] = mhlo.add %[[TMP_21]], %[[TMP_3]]
  // CHECK: %[[TMP_25:.*]] = mhlo.power %[[TMP_24]], %[[TMP_4]]
  // CHECK: %[[TMP_26:.*]] = mhlo.add %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.add %[[TMP_24]], %[[TMP_3]]
  // CHECK: %[[TMP_28:.*]] = mhlo.power %[[TMP_27]], %[[TMP_4]]
  // CHECK: %[[TMP_29:.*]] = mhlo.add %[[TMP_26]], %[[TMP_28]]
  // CHECK: %[[TMP_30:.*]] = mhlo.add %[[TMP_27]], %[[TMP_3]]
  // CHECK: %[[TMP_31:.*]] = mhlo.power %[[TMP_30]], %[[TMP_4]]
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_29]], %[[TMP_31]]
  // CHECK: %[[TMP_33:.*]] = mhlo.add %[[TMP_30]], %[[TMP_3]]
  // CHECK: %[[TMP_34:.*]] = mhlo.power %[[TMP_33]], %[[TMP_4]]
  // CHECK: %[[TMP_35:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_36:.*]] = mhlo.multiply %[[TMP_34]], %[[TMP_33]]
  // CHECK-DAG: %[[TMP_37:.*]] = mhlo.subtract %[[TMP_0]], %[[TMP_35]]
  // CHECK: %[[TMP_38:.*]] = mhlo.divide %[[TMP_36]], %[[TMP_37]]
  // CHECK: %[[TMP_39:.*]] = mhlo.multiply %[[TMP_33]], %[[TMP_33]]
  // CHECK: %[[TMP_40:.*]] = mhlo.divide %[[TMP_3]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = mhlo.constant dense<2.200000e+01>
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_0]], %[[TMP_41]]
  // CHECK: %[[TMP_43:.*]] = mhlo.constant dense<2.100000e+01>
  // CHECK: %[[TMP_44:.*]] = mhlo.add %[[TMP_0]], %[[TMP_43]]
  // CHECK: %[[TMP_45:.*]] = mhlo.multiply %[[TMP_42]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.constant dense<-1.39544646E-19>
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_2]], %[[TMP_46]]
  // CHECK: %[[TMP_48:.*]] = mhlo.multiply %[[TMP_40]], %[[TMP_47]]
  // CHECK: %[[TMP_49:.*]] = mhlo.multiply %[[TMP_45]], %[[TMP_48]]
  // CHECK: %[[TMP_50:.*]] = mhlo.constant dense<2.000000e+01>
  // CHECK: %[[TMP_51:.*]] = mhlo.add %[[TMP_0]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.constant dense<1.900000e+01>
  // CHECK: %[[TMP_53:.*]] = mhlo.add %[[TMP_0]], %[[TMP_52]]
  // CHECK: %[[TMP_54:.*]] = mhlo.multiply %[[TMP_51]], %[[TMP_53]]
  // CHECK: %[[TMP_55:.*]] = mhlo.constant dense<5.50900303E-18>
  // CHECK: %[[TMP_56:.*]] = mhlo.add %[[TMP_49]], %[[TMP_55]]
  // CHECK: %[[TMP_57:.*]] = mhlo.multiply %[[TMP_40]], %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = mhlo.multiply %[[TMP_54]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = mhlo.constant dense<1.800000e+01>
  // CHECK: %[[TMP_60:.*]] = mhlo.add %[[TMP_0]], %[[TMP_59]]
  // CHECK: %[[TMP_61:.*]] = mhlo.constant dense<1.700000e+01>
  // CHECK: %[[TMP_62:.*]] = mhlo.add %[[TMP_0]], %[[TMP_61]]
  // CHECK: %[[TMP_63:.*]] = mhlo.multiply %[[TMP_60]], %[[TMP_62]]
  // CHECK: %[[TMP_64:.*]] = mhlo.constant dense<-2.17486866E-16>
  // CHECK: %[[TMP_65:.*]] = mhlo.add %[[TMP_58]], %[[TMP_64]]
  // CHECK: %[[TMP_66:.*]] = mhlo.multiply %[[TMP_40]], %[[TMP_65]]
  // CHECK: %[[TMP_67:.*]] = mhlo.multiply %[[TMP_63]], %[[TMP_66]]
  // CHECK: %[[TMP_68:.*]] = mhlo.constant dense<1.600000e+01>
  // CHECK: %[[TMP_69:.*]] = mhlo.add %[[TMP_0]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = mhlo.constant dense<1.500000e+01>
  // CHECK: %[[TMP_71:.*]] = mhlo.add %[[TMP_0]], %[[TMP_70]]
  // CHECK: %[[TMP_72:.*]] = mhlo.multiply %[[TMP_69]], %[[TMP_71]]
  // CHECK: %[[TMP_73:.*]] = mhlo.constant dense<8.58606213E-15>
  // CHECK: %[[TMP_74:.*]] = mhlo.add %[[TMP_67]], %[[TMP_73]]
  // CHECK: %[[TMP_75:.*]] = mhlo.multiply %[[TMP_40]], %[[TMP_74]]
  // CHECK: %[[TMP_76:.*]] = mhlo.multiply %[[TMP_72]], %[[TMP_75]]
  // CHECK: %[[TMP_77:.*]] = mhlo.constant dense<1.400000e+01>
  // CHECK: %[[TMP_78:.*]] = mhlo.add %[[TMP_0]], %[[TMP_77]]
  // CHECK: %[[TMP_79:.*]] = mhlo.constant dense<1.300000e+01>
  // CHECK: %[[TMP_80:.*]] = mhlo.add %[[TMP_0]], %[[TMP_79]]
  // CHECK: %[[TMP_81:.*]] = mhlo.multiply %[[TMP_78]], %[[TMP_80]]
  // CHECK: %[[TMP_82:.*]] = mhlo.constant dense<-3.3896803E-13>
  // CHECK: %[[TMP_83:.*]] = mhlo.add %[[TMP_76]], %[[TMP_82]]
  // CHECK: %[[TMP_84:.*]] = mhlo.multiply %[[TMP_40]], %[[TMP_83]]
  // CHECK: %[[TMP_85:.*]] = mhlo.multiply %[[TMP_81]], %[[TMP_84]]
  // CHECK: %[[TMP_86:.*]] = mhlo.constant dense<1.200000e+01>
  // CHECK: %[[TMP_87:.*]] = mhlo.add %[[TMP_0]], %[[TMP_86]]
  // CHECK: %[[TMP_88:.*]] = mhlo.constant dense<1.100000e+01>
  // CHECK: %[[TMP_89:.*]] = mhlo.add %[[TMP_0]], %[[TMP_88]]
  // CHECK: %[[TMP_90:.*]] = mhlo.multiply %[[TMP_87]], %[[TMP_89]]
  // CHECK: %[[TMP_91:.*]] = mhlo.constant dense<1.33825364E-11>
  // CHECK: %[[TMP_92:.*]] = mhlo.add %[[TMP_85]], %[[TMP_91]]
  // CHECK: %[[TMP_93:.*]] = mhlo.multiply %[[TMP_40]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = mhlo.multiply %[[TMP_90]], %[[TMP_93]]
  // CHECK: %[[TMP_95:.*]] = mhlo.constant dense<1.000000e+01>
  // CHECK: %[[TMP_96:.*]] = mhlo.add %[[TMP_0]], %[[TMP_95]]
  // CHECK: %[[TMP_97:.*]] = mhlo.constant dense<9.000000e+00>
  // CHECK: %[[TMP_98:.*]] = mhlo.add %[[TMP_0]], %[[TMP_97]]
  // CHECK: %[[TMP_99:.*]] = mhlo.multiply %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.constant dense<-5.28419031E-10>
  // CHECK: %[[TMP_101:.*]] = mhlo.add %[[TMP_94]], %[[TMP_100]]
  // CHECK: %[[TMP_102:.*]] = mhlo.multiply %[[TMP_40]], %[[TMP_101]]
  // CHECK: %[[TMP_103:.*]] = mhlo.multiply %[[TMP_99]], %[[TMP_102]]
  // CHECK: %[[TMP_104:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_105:.*]] = mhlo.add %[[TMP_0]], %[[TMP_104]]
  // CHECK: %[[TMP_106:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_107:.*]] = mhlo.add %[[TMP_0]], %[[TMP_106]]
  // CHECK: %[[TMP_108:.*]] = mhlo.multiply %[[TMP_105]], %[[TMP_107]]
  // CHECK: %[[TMP_109:.*]] = mhlo.constant dense<2.08767563E-8>
  // CHECK: %[[TMP_110:.*]] = mhlo.add %[[TMP_103]], %[[TMP_109]]
  // CHECK: %[[TMP_111:.*]] = mhlo.multiply %[[TMP_40]], %[[TMP_110]]
  // CHECK: %[[TMP_112:.*]] = mhlo.multiply %[[TMP_108]], %[[TMP_111]]
  // CHECK: %[[TMP_113:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_114:.*]] = mhlo.add %[[TMP_0]], %[[TMP_113]]
  // CHECK: %[[TMP_115:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_116:.*]] = mhlo.add %[[TMP_0]], %[[TMP_115]]
  // CHECK: %[[TMP_117:.*]] = mhlo.multiply %[[TMP_114]], %[[TMP_116]]
  // CHECK: %[[TMP_118:.*]] = mhlo.constant dense<-8.26719599E-7>
  // CHECK: %[[TMP_119:.*]] = mhlo.add %[[TMP_112]], %[[TMP_118]]
  // CHECK: %[[TMP_120:.*]] = mhlo.multiply %[[TMP_40]], %[[TMP_119]]
  // CHECK: %[[TMP_121:.*]] = mhlo.multiply %[[TMP_117]], %[[TMP_120]]
  // CHECK: %[[TMP_122:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_123:.*]] = mhlo.add %[[TMP_0]], %[[TMP_122]]
  // CHECK: %[[TMP_124:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_125:.*]] = mhlo.add %[[TMP_0]], %[[TMP_124]]
  // CHECK: %[[TMP_126:.*]] = mhlo.multiply %[[TMP_123]], %[[TMP_125]]
  // CHECK: %[[TMP_127:.*]] = mhlo.constant dense<3.30687835E-5>
  // CHECK: %[[TMP_128:.*]] = mhlo.add %[[TMP_121]], %[[TMP_127]]
  // CHECK: %[[TMP_129:.*]] = mhlo.multiply %[[TMP_40]], %[[TMP_128]]
  // CHECK: %[[TMP_130:.*]] = mhlo.multiply %[[TMP_126]], %[[TMP_129]]
  // CHECK: %[[TMP_131:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_132:.*]] = mhlo.add %[[TMP_0]], %[[TMP_131]]
  // CHECK: %[[TMP_133:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_134:.*]] = mhlo.add %[[TMP_0]], %[[TMP_133]]
  // CHECK: %[[TMP_135:.*]] = mhlo.multiply %[[TMP_132]], %[[TMP_134]]
  // CHECK: %[[TMP_136:.*]] = mhlo.constant dense<-0.00138888892>
  // CHECK: %[[TMP_137:.*]] = mhlo.add %[[TMP_130]], %[[TMP_136]]
  // CHECK: %[[TMP_138:.*]] = mhlo.multiply %[[TMP_40]], %[[TMP_137]]
  // CHECK: %[[TMP_139:.*]] = mhlo.multiply %[[TMP_135]], %[[TMP_138]]
  // CHECK: %[[TMP_140:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_141:.*]] = mhlo.divide %[[TMP_0]], %[[TMP_33]]
  // CHECK: %[[TMP_142:.*]] = mhlo.constant dense<0.0833333358>
  // CHECK: %[[TMP_143:.*]] = mhlo.add %[[TMP_142]], %[[TMP_139]]
  // CHECK: %[[TMP_144:.*]] = mhlo.multiply %[[TMP_141]], %[[TMP_143]]
  // CHECK: %[[TMP_145:.*]] = mhlo.add %[[TMP_140]], %[[TMP_144]]
  // CHECK: %[[TMP_146:.*]] = mhlo.multiply %[[TMP_34]], %[[TMP_145]]
  // CHECK: %[[TMP_147:.*]] = mhlo.add %[[TMP_32]], %[[TMP_38]]
  // CHECK: %[[TMP_148:.*]] = mhlo.add %[[TMP_147]], %[[TMP_146]]
  // CHECK: %[[TMP_149:.*]] = mhlo.abs %[[TMP_34]]
  // CHECK: %[[TMP_150:.*]] = mhlo.abs %[[TMP_32]]
  // CHECK: %[[TMP_151:.*]] = mhlo.constant dense<1.401300e-45>
  // CHECK: %[[TMP_152:.*]] = mhlo.multiply %[[TMP_150]], %[[TMP_151]]
  // CHECK: %[[TMP_153:.*]] = mhlo.compare LT, %[[TMP_149]], %[[TMP_152]]
  // CHECK: %[[TMP_154:.*]] = mhlo.select %[[TMP_153]], %[[TMP_32]], %[[TMP_148]]
  // CHECK: %[[TMP_155:.*]] = mhlo.constant dense<0x7FC00000>
  // CHECK: %[[TMP_156:.*]] = mhlo.compare LT, %[[TMP_0]], %[[TMP_35]]
  // CHECK: %[[TMP_157:.*]] = mhlo.select %[[TMP_156]], %[[TMP_155]], %[[TMP_154]]
  // CHECK: %[[TMP_158:.*]] = mhlo.compare LE, %[[TMP_1]], %[[TMP_2]]
  // CHECK: %[[TMP_159:.*]] = mhlo.floor %[[TMP_0]]
  // CHECK: %[[TMP_160:.*]] = mhlo.compare NE, %[[TMP_0]], %[[TMP_159]]
  // CHECK: %[[TMP_161:.*]] = mhlo.and %[[TMP_158]], %[[TMP_160]] : tensor<i1>
  // CHECK: %[[TMP_162:.*]] = mhlo.select %[[TMP_161]], %[[TMP_155]], %[[TMP_157]]
  // CHECK: %[[TMP_163:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_164:.*]] = mhlo.floor %[[TMP_1]]
  // CHECK: %[[TMP_165:.*]] = mhlo.compare EQ, %[[TMP_1]], %[[TMP_164]]
  // CHECK: %[[TMP_166:.*]] = mhlo.and %[[TMP_158]], %[[TMP_165]] : tensor<i1>
  // CHECK: %[[TMP_167:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_168:.*]] = mhlo.floor %[[TMP_0]]
  // CHECK: %[[TMP_169:.*]] = mhlo.compare EQ, %[[TMP_0]], %[[TMP_168]]
  // CHECK: %[[TMP_170:.*]] = mhlo.remainder %[[TMP_0]], %[[TMP_167]]
  // CHECK: %[[TMP_171:.*]] = mhlo.compare EQ, %[[TMP_170]], %[[TMP_2]]
  // CHECK: %[[TMP_172:.*]] = mhlo.and %[[TMP_169]], %[[TMP_171]] : tensor<i1>
  // CHECK: %[[TMP_173:.*]] = mhlo.select %[[TMP_172]], %[[TMP_163]], %[[TMP_155]]
  // CHECK: %[[TMP_174:.*]] = mhlo.select %[[TMP_166]], %[[TMP_173]], %[[TMP_162]]
  // CHECK: %[[TMP_175:.*]] = mhlo.compare EQ, %[[TMP_0]], %[[TMP_3]]
  // CHECK: %[[TMP_176:.*]] = mhlo.select %[[TMP_175]], %[[TMP_163]], %[[TMP_174]]
  // CHECK: %[[TMP_177:.*]] = mhlo.convert %[[TMP_176]] : (tensor<f32>) -> tensor<f16>
  %0 = chlo.zeta %arg0, %arg1 : tensor<f16>, tensor<f16> -> tensor<f16>
  func.return %0 : tensor<f16>
}

// -----

// CHECK: @polygamma_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>)
func.func @polygamma_f32(%lhs : tensor<f32>, %rhs : tensor<f32>) -> tensor<f32> {
  // CHECK-DAG: %[[TMP_0:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_1:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_2:.*]] = mhlo.remainder %[[ARG0]], %[[TMP_1]]
  // CHECK: %[[TMP_3:.*]] = mhlo.multiply %[[TMP_1]], %[[TMP_2]]
  // CHECK: %[[TMP_4:.*]] = mhlo.subtract %[[TMP_3]], %[[TMP_0]]
  // CHECK: %[[TMP_5:.*]] = mhlo.add %[[ARG0]], %[[TMP_0]]
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_7:.*]] = mhlo.compare LT, %[[TMP_5]], %[[TMP_6]]
  // CHECK: %[[TMP_8:.*]] = mhlo.negate %[[TMP_5]]
  // CHECK: %[[TMP_9:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = mhlo.select %[[TMP_7]], %[[TMP_8]], %[[TMP_10]]
  // CHECK-DAG: %[[TMP_12:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_13:.*]] = mhlo.constant dense<676.520386>
  // CHECK-DAG: %[[TMP_14:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_11]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = mhlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = mhlo.add %[[TMP_12]], %[[TMP_16]]
  // CHECK-DAG: %[[TMP_18:.*]] = mhlo.constant dense<-1259.13916>
  // CHECK-DAG: %[[TMP_19:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_11]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK-DAG: %[[TMP_23:.*]] = mhlo.constant dense<771.323425>
  // CHECK-DAG: %[[TMP_24:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = mhlo.add %[[TMP_11]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = mhlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK-DAG: %[[TMP_28:.*]] = mhlo.constant dense<-176.615036>
  // CHECK-DAG: %[[TMP_29:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = mhlo.add %[[TMP_11]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = mhlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK-DAG: %[[TMP_33:.*]] = mhlo.constant dense<12.5073433>
  // CHECK-DAG: %[[TMP_34:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = mhlo.add %[[TMP_11]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK-DAG: %[[TMP_38:.*]] = mhlo.constant dense<-0.138571098>
  // CHECK-DAG: %[[TMP_39:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = mhlo.add %[[TMP_11]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = mhlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK-DAG: %[[TMP_43:.*]] = mhlo.constant dense<9.98436917E-6>
  // CHECK-DAG: %[[TMP_44:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = mhlo.add %[[TMP_11]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = mhlo.constant dense<1.50563267E-7>
  // CHECK-DAG: %[[TMP_49:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_11]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_54:.*]] = mhlo.add %[[TMP_53]], %[[TMP_11]]
  // CHECK: %[[TMP_55:.*]] = mhlo.constant dense<2.01490307>
  // CHECK: %[[TMP_56:.*]] = mhlo.divide %[[TMP_11]], %[[TMP_53]]
  // CHECK: %[[TMP_57:.*]] = mhlo.log_plus_one %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_55]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = mhlo.divide %[[TMP_54]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = mhlo.add %[[TMP_11]], %[[TMP_6]]
  // CHECK: %[[TMP_61:.*]] = mhlo.subtract %[[TMP_60]], %[[TMP_59]]
  // CHECK: %[[TMP_62:.*]] = mhlo.multiply %[[TMP_61]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = mhlo.log %[[TMP_52]]
  // CHECK: %[[TMP_64:.*]] = mhlo.constant dense<0.918938517>
  // CHECK: %[[TMP_65:.*]] = mhlo.add %[[TMP_64]], %[[TMP_62]]
  // CHECK: %[[TMP_66:.*]] = mhlo.add %[[TMP_65]], %[[TMP_63]]
  // CHECK: %[[TMP_67:.*]] = mhlo.abs %[[TMP_5]]
  // CHECK: %[[TMP_68:.*]] = mhlo.floor %[[TMP_67]]
  // CHECK: %[[TMP_69:.*]] = mhlo.subtract %[[TMP_67]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = mhlo.compare LT, %[[TMP_6]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = mhlo.subtract %[[TMP_9]], %[[TMP_69]]
  // CHECK: %[[TMP_72:.*]] = mhlo.select %[[TMP_70]], %[[TMP_71]], %[[TMP_69]]
  // CHECK: %[[TMP_73:.*]] = mhlo.constant dense<3.14159274>
  // CHECK: %[[TMP_74:.*]] = mhlo.multiply %[[TMP_73]], %[[TMP_72]]
  // CHECK: %[[TMP_75:.*]] = mhlo.sine %[[TMP_74]]
  // CHECK: %[[TMP_76:.*]] = mhlo.log %[[TMP_75]]
  // CHECK: %[[TMP_77:.*]] = mhlo.constant dense<1.14472985>
  // CHECK: %[[TMP_78:.*]] = mhlo.subtract %[[TMP_77]], %[[TMP_76]]
  // CHECK: %[[TMP_79:.*]] = mhlo.subtract %[[TMP_78]], %[[TMP_66]]
  // CHECK: %[[TMP_80:.*]] = mhlo.is_finite %[[TMP_76]]
  // CHECK: %[[TMP_81:.*]] = mhlo.negate %[[TMP_76]]
  // CHECK: %[[TMP_82:.*]] = mhlo.select %[[TMP_80]], %[[TMP_79]], %[[TMP_81]]
  // CHECK: %[[TMP_83:.*]] = mhlo.select %[[TMP_7]], %[[TMP_82]], %[[TMP_66]]
  // CHECK: %[[TMP_84:.*]] = mhlo.abs %[[TMP_5]]
  // CHECK: %[[TMP_85:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_86:.*]] = mhlo.compare EQ, %[[TMP_84]], %[[TMP_85]]
  // CHECK: %[[TMP_87:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_88:.*]] = mhlo.select %[[TMP_86]], %[[TMP_87]], %[[TMP_83]]
  // CHECK: %[[TMP_89:.*]] = mhlo.exponential %[[TMP_88]]
  // CHECK-DAG: %[[TMP_90:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_91:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_92:.*]] = mhlo.negate %[[TMP_5]]
  // CHECK: %[[TMP_93:.*]] = mhlo.power %[[ARG1]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = mhlo.add %[[ARG1]], %[[TMP_91]]
  // CHECK: %[[TMP_95:.*]] = mhlo.power %[[TMP_94]], %[[TMP_92]]
  // CHECK: %[[TMP_96:.*]] = mhlo.add %[[TMP_93]], %[[TMP_95]]
  // CHECK: %[[TMP_97:.*]] = mhlo.add %[[TMP_94]], %[[TMP_91]]
  // CHECK: %[[TMP_98:.*]] = mhlo.power %[[TMP_97]], %[[TMP_92]]
  // CHECK: %[[TMP_99:.*]] = mhlo.add %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.add %[[TMP_97]], %[[TMP_91]]
  // CHECK: %[[TMP_101:.*]] = mhlo.power %[[TMP_100]], %[[TMP_92]]
  // CHECK: %[[TMP_102:.*]] = mhlo.add %[[TMP_99]], %[[TMP_101]]
  // CHECK: %[[TMP_103:.*]] = mhlo.add %[[TMP_100]], %[[TMP_91]]
  // CHECK: %[[TMP_104:.*]] = mhlo.power %[[TMP_103]], %[[TMP_92]]
  // CHECK: %[[TMP_105:.*]] = mhlo.add %[[TMP_102]], %[[TMP_104]]
  // CHECK: %[[TMP_106:.*]] = mhlo.add %[[TMP_103]], %[[TMP_91]]
  // CHECK: %[[TMP_107:.*]] = mhlo.power %[[TMP_106]], %[[TMP_92]]
  // CHECK: %[[TMP_108:.*]] = mhlo.add %[[TMP_105]], %[[TMP_107]]
  // CHECK: %[[TMP_109:.*]] = mhlo.add %[[TMP_106]], %[[TMP_91]]
  // CHECK: %[[TMP_110:.*]] = mhlo.power %[[TMP_109]], %[[TMP_92]]
  // CHECK: %[[TMP_111:.*]] = mhlo.add %[[TMP_108]], %[[TMP_110]]
  // CHECK: %[[TMP_112:.*]] = mhlo.add %[[TMP_109]], %[[TMP_91]]
  // CHECK: %[[TMP_113:.*]] = mhlo.power %[[TMP_112]], %[[TMP_92]]
  // CHECK: %[[TMP_114:.*]] = mhlo.add %[[TMP_111]], %[[TMP_113]]
  // CHECK: %[[TMP_115:.*]] = mhlo.add %[[TMP_112]], %[[TMP_91]]
  // CHECK: %[[TMP_116:.*]] = mhlo.power %[[TMP_115]], %[[TMP_92]]
  // CHECK: %[[TMP_117:.*]] = mhlo.add %[[TMP_114]], %[[TMP_116]]
  // CHECK: %[[TMP_118:.*]] = mhlo.add %[[TMP_115]], %[[TMP_91]]
  // CHECK: %[[TMP_119:.*]] = mhlo.power %[[TMP_118]], %[[TMP_92]]
  // CHECK: %[[TMP_120:.*]] = mhlo.add %[[TMP_117]], %[[TMP_119]]
  // CHECK: %[[TMP_121:.*]] = mhlo.add %[[TMP_118]], %[[TMP_91]]
  // CHECK: %[[TMP_122:.*]] = mhlo.power %[[TMP_121]], %[[TMP_92]]
  // CHECK: %[[TMP_123:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_124:.*]] = mhlo.multiply %[[TMP_122]], %[[TMP_121]]
  // CHECK-DAG: %[[TMP_125:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_123]]
  // CHECK: %[[TMP_126:.*]] = mhlo.divide %[[TMP_124]], %[[TMP_125]]
  // CHECK: %[[TMP_127:.*]] = mhlo.multiply %[[TMP_121]], %[[TMP_121]]
  // CHECK: %[[TMP_128:.*]] = mhlo.divide %[[TMP_91]], %[[TMP_127]]
  // CHECK: %[[TMP_129:.*]] = mhlo.constant dense<2.200000e+01>
  // CHECK: %[[TMP_130:.*]] = mhlo.add %[[TMP_5]], %[[TMP_129]]
  // CHECK: %[[TMP_131:.*]] = mhlo.constant dense<2.100000e+01>
  // CHECK: %[[TMP_132:.*]] = mhlo.add %[[TMP_5]], %[[TMP_131]]
  // CHECK: %[[TMP_133:.*]] = mhlo.multiply %[[TMP_130]], %[[TMP_132]]
  // CHECK: %[[TMP_134:.*]] = mhlo.constant dense<-1.39544646E-19>
  // CHECK: %[[TMP_135:.*]] = mhlo.add %[[TMP_90]], %[[TMP_134]]
  // CHECK: %[[TMP_136:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_135]]
  // CHECK: %[[TMP_137:.*]] = mhlo.multiply %[[TMP_133]], %[[TMP_136]]
  // CHECK: %[[TMP_138:.*]] = mhlo.constant dense<2.000000e+01>
  // CHECK: %[[TMP_139:.*]] = mhlo.add %[[TMP_5]], %[[TMP_138]]
  // CHECK: %[[TMP_140:.*]] = mhlo.constant dense<1.900000e+01>
  // CHECK: %[[TMP_141:.*]] = mhlo.add %[[TMP_5]], %[[TMP_140]]
  // CHECK: %[[TMP_142:.*]] = mhlo.multiply %[[TMP_139]], %[[TMP_141]]
  // CHECK: %[[TMP_143:.*]] = mhlo.constant dense<5.50900303E-18>
  // CHECK: %[[TMP_144:.*]] = mhlo.add %[[TMP_137]], %[[TMP_143]]
  // CHECK: %[[TMP_145:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_144]]
  // CHECK: %[[TMP_146:.*]] = mhlo.multiply %[[TMP_142]], %[[TMP_145]]
  // CHECK: %[[TMP_147:.*]] = mhlo.constant dense<1.800000e+01>
  // CHECK: %[[TMP_148:.*]] = mhlo.add %[[TMP_5]], %[[TMP_147]]
  // CHECK: %[[TMP_149:.*]] = mhlo.constant dense<1.700000e+01>
  // CHECK: %[[TMP_150:.*]] = mhlo.add %[[TMP_5]], %[[TMP_149]]
  // CHECK: %[[TMP_151:.*]] = mhlo.multiply %[[TMP_148]], %[[TMP_150]]
  // CHECK: %[[TMP_152:.*]] = mhlo.constant dense<-2.17486866E-16>
  // CHECK: %[[TMP_153:.*]] = mhlo.add %[[TMP_146]], %[[TMP_152]]
  // CHECK: %[[TMP_154:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_153]]
  // CHECK: %[[TMP_155:.*]] = mhlo.multiply %[[TMP_151]], %[[TMP_154]]
  // CHECK: %[[TMP_156:.*]] = mhlo.constant dense<1.600000e+01>
  // CHECK: %[[TMP_157:.*]] = mhlo.add %[[TMP_5]], %[[TMP_156]]
  // CHECK: %[[TMP_158:.*]] = mhlo.constant dense<1.500000e+01>
  // CHECK: %[[TMP_159:.*]] = mhlo.add %[[TMP_5]], %[[TMP_158]]
  // CHECK: %[[TMP_160:.*]] = mhlo.multiply %[[TMP_157]], %[[TMP_159]]
  // CHECK: %[[TMP_161:.*]] = mhlo.constant dense<8.58606213E-15>
  // CHECK: %[[TMP_162:.*]] = mhlo.add %[[TMP_155]], %[[TMP_161]]
  // CHECK: %[[TMP_163:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_162]]
  // CHECK: %[[TMP_164:.*]] = mhlo.multiply %[[TMP_160]], %[[TMP_163]]
  // CHECK: %[[TMP_165:.*]] = mhlo.constant dense<1.400000e+01>
  // CHECK: %[[TMP_166:.*]] = mhlo.add %[[TMP_5]], %[[TMP_165]]
  // CHECK: %[[TMP_167:.*]] = mhlo.constant dense<1.300000e+01>
  // CHECK: %[[TMP_168:.*]] = mhlo.add %[[TMP_5]], %[[TMP_167]]
  // CHECK: %[[TMP_169:.*]] = mhlo.multiply %[[TMP_166]], %[[TMP_168]]
  // CHECK: %[[TMP_170:.*]] = mhlo.constant dense<-3.3896803E-13>
  // CHECK: %[[TMP_171:.*]] = mhlo.add %[[TMP_164]], %[[TMP_170]]
  // CHECK: %[[TMP_172:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_171]]
  // CHECK: %[[TMP_173:.*]] = mhlo.multiply %[[TMP_169]], %[[TMP_172]]
  // CHECK: %[[TMP_174:.*]] = mhlo.constant dense<1.200000e+01>
  // CHECK: %[[TMP_175:.*]] = mhlo.add %[[TMP_5]], %[[TMP_174]]
  // CHECK: %[[TMP_176:.*]] = mhlo.constant dense<1.100000e+01>
  // CHECK: %[[TMP_177:.*]] = mhlo.add %[[TMP_5]], %[[TMP_176]]
  // CHECK: %[[TMP_178:.*]] = mhlo.multiply %[[TMP_175]], %[[TMP_177]]
  // CHECK: %[[TMP_179:.*]] = mhlo.constant dense<1.33825364E-11>
  // CHECK: %[[TMP_180:.*]] = mhlo.add %[[TMP_173]], %[[TMP_179]]
  // CHECK: %[[TMP_181:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_180]]
  // CHECK: %[[TMP_182:.*]] = mhlo.multiply %[[TMP_178]], %[[TMP_181]]
  // CHECK: %[[TMP_183:.*]] = mhlo.constant dense<1.000000e+01>
  // CHECK: %[[TMP_184:.*]] = mhlo.add %[[TMP_5]], %[[TMP_183]]
  // CHECK: %[[TMP_185:.*]] = mhlo.constant dense<9.000000e+00>
  // CHECK: %[[TMP_186:.*]] = mhlo.add %[[TMP_5]], %[[TMP_185]]
  // CHECK: %[[TMP_187:.*]] = mhlo.multiply %[[TMP_184]], %[[TMP_186]]
  // CHECK: %[[TMP_188:.*]] = mhlo.constant dense<-5.28419031E-10>
  // CHECK: %[[TMP_189:.*]] = mhlo.add %[[TMP_182]], %[[TMP_188]]
  // CHECK: %[[TMP_190:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_189]]
  // CHECK: %[[TMP_191:.*]] = mhlo.multiply %[[TMP_187]], %[[TMP_190]]
  // CHECK: %[[TMP_192:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_193:.*]] = mhlo.add %[[TMP_5]], %[[TMP_192]]
  // CHECK: %[[TMP_194:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_195:.*]] = mhlo.add %[[TMP_5]], %[[TMP_194]]
  // CHECK: %[[TMP_196:.*]] = mhlo.multiply %[[TMP_193]], %[[TMP_195]]
  // CHECK: %[[TMP_197:.*]] = mhlo.constant dense<2.08767563E-8>
  // CHECK: %[[TMP_198:.*]] = mhlo.add %[[TMP_191]], %[[TMP_197]]
  // CHECK: %[[TMP_199:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_198]]
  // CHECK: %[[TMP_200:.*]] = mhlo.multiply %[[TMP_196]], %[[TMP_199]]
  // CHECK: %[[TMP_201:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_202:.*]] = mhlo.add %[[TMP_5]], %[[TMP_201]]
  // CHECK: %[[TMP_203:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_204:.*]] = mhlo.add %[[TMP_5]], %[[TMP_203]]
  // CHECK: %[[TMP_205:.*]] = mhlo.multiply %[[TMP_202]], %[[TMP_204]]
  // CHECK: %[[TMP_206:.*]] = mhlo.constant dense<-8.26719599E-7>
  // CHECK: %[[TMP_207:.*]] = mhlo.add %[[TMP_200]], %[[TMP_206]]
  // CHECK: %[[TMP_208:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_207]]
  // CHECK: %[[TMP_209:.*]] = mhlo.multiply %[[TMP_205]], %[[TMP_208]]
  // CHECK: %[[TMP_210:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_211:.*]] = mhlo.add %[[TMP_5]], %[[TMP_210]]
  // CHECK: %[[TMP_212:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_213:.*]] = mhlo.add %[[TMP_5]], %[[TMP_212]]
  // CHECK: %[[TMP_214:.*]] = mhlo.multiply %[[TMP_211]], %[[TMP_213]]
  // CHECK: %[[TMP_215:.*]] = mhlo.constant dense<3.30687835E-5>
  // CHECK: %[[TMP_216:.*]] = mhlo.add %[[TMP_209]], %[[TMP_215]]
  // CHECK: %[[TMP_217:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_216]]
  // CHECK: %[[TMP_218:.*]] = mhlo.multiply %[[TMP_214]], %[[TMP_217]]
  // CHECK: %[[TMP_219:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_220:.*]] = mhlo.add %[[TMP_5]], %[[TMP_219]]
  // CHECK: %[[TMP_221:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_222:.*]] = mhlo.add %[[TMP_5]], %[[TMP_221]]
  // CHECK: %[[TMP_223:.*]] = mhlo.multiply %[[TMP_220]], %[[TMP_222]]
  // CHECK: %[[TMP_224:.*]] = mhlo.constant dense<-0.00138888892>
  // CHECK: %[[TMP_225:.*]] = mhlo.add %[[TMP_218]], %[[TMP_224]]
  // CHECK: %[[TMP_226:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_225]]
  // CHECK: %[[TMP_227:.*]] = mhlo.multiply %[[TMP_223]], %[[TMP_226]]
  // CHECK: %[[TMP_228:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_229:.*]] = mhlo.divide %[[TMP_5]], %[[TMP_121]]
  // CHECK: %[[TMP_230:.*]] = mhlo.constant dense<0.0833333358>
  // CHECK: %[[TMP_231:.*]] = mhlo.add %[[TMP_230]], %[[TMP_227]]
  // CHECK: %[[TMP_232:.*]] = mhlo.multiply %[[TMP_229]], %[[TMP_231]]
  // CHECK: %[[TMP_233:.*]] = mhlo.add %[[TMP_228]], %[[TMP_232]]
  // CHECK: %[[TMP_234:.*]] = mhlo.multiply %[[TMP_122]], %[[TMP_233]]
  // CHECK: %[[TMP_235:.*]] = mhlo.add %[[TMP_120]], %[[TMP_126]]
  // CHECK: %[[TMP_236:.*]] = mhlo.add %[[TMP_235]], %[[TMP_234]]
  // CHECK: %[[TMP_237:.*]] = mhlo.abs %[[TMP_122]]
  // CHECK: %[[TMP_238:.*]] = mhlo.abs %[[TMP_120]]
  // CHECK: %[[TMP_239:.*]] = mhlo.constant dense<1.401300e-45>
  // CHECK: %[[TMP_240:.*]] = mhlo.multiply %[[TMP_238]], %[[TMP_239]]
  // CHECK: %[[TMP_241:.*]] = mhlo.compare LT, %[[TMP_237]], %[[TMP_240]]
  // CHECK: %[[TMP_242:.*]] = mhlo.select %[[TMP_241]], %[[TMP_120]], %[[TMP_236]]
  // CHECK: %[[TMP_243:.*]] = mhlo.constant dense<0x7FC00000>
  // CHECK: %[[TMP_244:.*]] = mhlo.compare LT, %[[TMP_5]], %[[TMP_123]]
  // CHECK: %[[TMP_245:.*]] = mhlo.select %[[TMP_244]], %[[TMP_243]], %[[TMP_242]]
  // CHECK: %[[TMP_246:.*]] = mhlo.compare LE, %[[ARG1]], %[[TMP_90]]
  // CHECK: %[[TMP_247:.*]] = mhlo.floor %[[TMP_5]]
  // CHECK: %[[TMP_248:.*]] = mhlo.compare NE, %[[TMP_5]], %[[TMP_247]]
  // CHECK: %[[TMP_249:.*]] = mhlo.and %[[TMP_246]], %[[TMP_248]]
  // CHECK: %[[TMP_250:.*]] = mhlo.select %[[TMP_249]], %[[TMP_243]], %[[TMP_245]]
  // CHECK: %[[TMP_251:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK: %[[TMP_252:.*]] = mhlo.floor %[[ARG1]]
  // CHECK: %[[TMP_253:.*]] = mhlo.compare EQ, %[[ARG1]], %[[TMP_252]]
  // CHECK: %[[TMP_254:.*]] = mhlo.and %[[TMP_246]], %[[TMP_253]]
  // CHECK: %[[TMP_255:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_256:.*]] = mhlo.floor %[[TMP_5]]
  // CHECK: %[[TMP_257:.*]] = mhlo.compare EQ, %[[TMP_5]], %[[TMP_256]]
  // CHECK: %[[TMP_258:.*]] = mhlo.remainder %[[TMP_5]], %[[TMP_255]]
  // CHECK: %[[TMP_259:.*]] = mhlo.compare EQ, %[[TMP_258]], %[[TMP_90]]
  // CHECK: %[[TMP_260:.*]] = mhlo.and %[[TMP_257]], %[[TMP_259]]
  // CHECK: %[[TMP_261:.*]] = mhlo.select %[[TMP_260]], %[[TMP_251]], %[[TMP_243]]
  // CHECK: %[[TMP_262:.*]] = mhlo.select %[[TMP_254]], %[[TMP_261]], %[[TMP_250]]
  // CHECK: %[[TMP_263:.*]] = mhlo.compare EQ, %[[TMP_5]], %[[TMP_91]]
  // CHECK: %[[TMP_264:.*]] = mhlo.select %[[TMP_263]], %[[TMP_251]], %[[TMP_262]]
  // CHECK: %[[TMP_265:.*]] = mhlo.multiply %[[TMP_4]], %[[TMP_89]]
  // CHECK: %[[TMP_266:.*]] = mhlo.multiply %[[TMP_265]], %[[TMP_264]]
  // CHECK: %[[TMP_267:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_268:.*]] = mhlo.compare EQ, %[[ARG0]], %[[TMP_267]]
  // CHECK: %[[TMP_269:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_270:.*]] = mhlo.compare LT, %[[ARG1]], %[[TMP_269]]
  // CHECK: %[[TMP_271:.*]] = mhlo.negate %[[ARG1]]
  // CHECK: %[[TMP_272:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_273:.*]] = mhlo.subtract %[[ARG1]], %[[TMP_272]]
  // CHECK: %[[TMP_274:.*]] = mhlo.select %[[TMP_270]], %[[TMP_271]], %[[TMP_273]]
  // CHECK-DAG: %[[TMP_275:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_276:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_277:.*]] = mhlo.constant dense<676.520386>
  // CHECK-DAG: %[[TMP_278:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_279:.*]] = mhlo.add %[[TMP_274]], %[[TMP_278]]
  // CHECK: %[[TMP_280:.*]] = mhlo.multiply %[[TMP_279]], %[[TMP_279]]
  // CHECK: %[[TMP_281:.*]] = mhlo.divide %[[TMP_277]], %[[TMP_280]]
  // CHECK: %[[TMP_282:.*]] = mhlo.subtract %[[TMP_275]], %[[TMP_281]]
  // CHECK: %[[TMP_283:.*]] = mhlo.divide %[[TMP_277]], %[[TMP_279]]
  // CHECK: %[[TMP_284:.*]] = mhlo.add %[[TMP_276]], %[[TMP_283]]
  // CHECK-DAG: %[[TMP_285:.*]] = mhlo.constant dense<-1259.13916>
  // CHECK-DAG: %[[TMP_286:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_287:.*]] = mhlo.add %[[TMP_274]], %[[TMP_286]]
  // CHECK: %[[TMP_288:.*]] = mhlo.multiply %[[TMP_287]], %[[TMP_287]]
  // CHECK: %[[TMP_289:.*]] = mhlo.divide %[[TMP_285]], %[[TMP_288]]
  // CHECK: %[[TMP_290:.*]] = mhlo.subtract %[[TMP_282]], %[[TMP_289]]
  // CHECK: %[[TMP_291:.*]] = mhlo.divide %[[TMP_285]], %[[TMP_287]]
  // CHECK: %[[TMP_292:.*]] = mhlo.add %[[TMP_284]], %[[TMP_291]]
  // CHECK-DAG: %[[TMP_293:.*]] = mhlo.constant dense<771.323425>
  // CHECK-DAG: %[[TMP_294:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_295:.*]] = mhlo.add %[[TMP_274]], %[[TMP_294]]
  // CHECK: %[[TMP_296:.*]] = mhlo.multiply %[[TMP_295]], %[[TMP_295]]
  // CHECK: %[[TMP_297:.*]] = mhlo.divide %[[TMP_293]], %[[TMP_296]]
  // CHECK: %[[TMP_298:.*]] = mhlo.subtract %[[TMP_290]], %[[TMP_297]]
  // CHECK: %[[TMP_299:.*]] = mhlo.divide %[[TMP_293]], %[[TMP_295]]
  // CHECK: %[[TMP_300:.*]] = mhlo.add %[[TMP_292]], %[[TMP_299]]
  // CHECK-DAG: %[[TMP_301:.*]] = mhlo.constant dense<-176.615036>
  // CHECK-DAG: %[[TMP_302:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_303:.*]] = mhlo.add %[[TMP_274]], %[[TMP_302]]
  // CHECK: %[[TMP_304:.*]] = mhlo.multiply %[[TMP_303]], %[[TMP_303]]
  // CHECK: %[[TMP_305:.*]] = mhlo.divide %[[TMP_301]], %[[TMP_304]]
  // CHECK: %[[TMP_306:.*]] = mhlo.subtract %[[TMP_298]], %[[TMP_305]]
  // CHECK: %[[TMP_307:.*]] = mhlo.divide %[[TMP_301]], %[[TMP_303]]
  // CHECK: %[[TMP_308:.*]] = mhlo.add %[[TMP_300]], %[[TMP_307]]
  // CHECK-DAG: %[[TMP_309:.*]] = mhlo.constant dense<12.5073433>
  // CHECK-DAG: %[[TMP_310:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_311:.*]] = mhlo.add %[[TMP_274]], %[[TMP_310]]
  // CHECK: %[[TMP_312:.*]] = mhlo.multiply %[[TMP_311]], %[[TMP_311]]
  // CHECK: %[[TMP_313:.*]] = mhlo.divide %[[TMP_309]], %[[TMP_312]]
  // CHECK: %[[TMP_314:.*]] = mhlo.subtract %[[TMP_306]], %[[TMP_313]]
  // CHECK: %[[TMP_315:.*]] = mhlo.divide %[[TMP_309]], %[[TMP_311]]
  // CHECK: %[[TMP_316:.*]] = mhlo.add %[[TMP_308]], %[[TMP_315]]
  // CHECK-DAG: %[[TMP_317:.*]] = mhlo.constant dense<-0.138571098>
  // CHECK-DAG: %[[TMP_318:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_319:.*]] = mhlo.add %[[TMP_274]], %[[TMP_318]]
  // CHECK: %[[TMP_320:.*]] = mhlo.multiply %[[TMP_319]], %[[TMP_319]]
  // CHECK: %[[TMP_321:.*]] = mhlo.divide %[[TMP_317]], %[[TMP_320]]
  // CHECK: %[[TMP_322:.*]] = mhlo.subtract %[[TMP_314]], %[[TMP_321]]
  // CHECK: %[[TMP_323:.*]] = mhlo.divide %[[TMP_317]], %[[TMP_319]]
  // CHECK: %[[TMP_324:.*]] = mhlo.add %[[TMP_316]], %[[TMP_323]]
  // CHECK-DAG: %[[TMP_325:.*]] = mhlo.constant dense<9.98436917E-6>
  // CHECK-DAG: %[[TMP_326:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_327:.*]] = mhlo.add %[[TMP_274]], %[[TMP_326]]
  // CHECK: %[[TMP_328:.*]] = mhlo.multiply %[[TMP_327]], %[[TMP_327]]
  // CHECK: %[[TMP_329:.*]] = mhlo.divide %[[TMP_325]], %[[TMP_328]]
  // CHECK: %[[TMP_330:.*]] = mhlo.subtract %[[TMP_322]], %[[TMP_329]]
  // CHECK: %[[TMP_331:.*]] = mhlo.divide %[[TMP_325]], %[[TMP_327]]
  // CHECK: %[[TMP_332:.*]] = mhlo.add %[[TMP_324]], %[[TMP_331]]
  // CHECK-DAG: %[[TMP_333:.*]] = mhlo.constant dense<1.50563267E-7>
  // CHECK-DAG: %[[TMP_334:.*]] = mhlo.constant dense<8.000000e+00>
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
  // CHECK: %[[TMP_345:.*]] = mhlo.log_plus_one %[[TMP_344]]
  // CHECK: %[[TMP_346:.*]] = mhlo.add %[[TMP_343]], %[[TMP_345]]
  // CHECK: %[[TMP_347:.*]] = mhlo.divide %[[TMP_338]], %[[TMP_340]]
  // CHECK: %[[TMP_348:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_349:.*]] = mhlo.divide %[[TMP_348]], %[[TMP_342]]
  // CHECK: %[[TMP_350:.*]] = mhlo.add %[[TMP_346]], %[[TMP_347]]
  // CHECK: %[[TMP_351:.*]] = mhlo.subtract %[[TMP_350]], %[[TMP_349]]
  // CHECK: %[[TMP_352:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_353:.*]] = mhlo.add %[[ARG1]], %[[TMP_352]]
  // CHECK: %[[TMP_354:.*]] = mhlo.floor %[[TMP_353]]
  // CHECK: %[[TMP_355:.*]] = mhlo.abs %[[TMP_354]]
  // CHECK: %[[TMP_356:.*]] = mhlo.add %[[ARG1]], %[[TMP_355]]
  // CHECK: %[[TMP_357:.*]] = mhlo.constant dense<3.14159274>
  // CHECK: %[[TMP_358:.*]] = mhlo.multiply %[[TMP_357]], %[[TMP_356]]
  // CHECK: %[[TMP_359:.*]] = mhlo.cosine %[[TMP_358]]
  // CHECK: %[[TMP_360:.*]] = mhlo.sine %[[TMP_358]]
  // CHECK: %[[TMP_361:.*]] = mhlo.multiply %[[TMP_357]], %[[TMP_359]]
  // CHECK: %[[TMP_362:.*]] = mhlo.divide %[[TMP_361]], %[[TMP_360]]
  // CHECK: %[[TMP_363:.*]] = mhlo.subtract %[[TMP_351]], %[[TMP_362]]
  // CHECK: %[[TMP_364:.*]] = mhlo.select %[[TMP_270]], %[[TMP_363]], %[[TMP_351]]
  // CHECK: %[[TMP_365:.*]] = mhlo.compare LE, %[[ARG1]], %[[TMP_275]]
  // CHECK: %[[TMP_366:.*]] = mhlo.floor %[[ARG1]]
  // CHECK: %[[TMP_367:.*]] = mhlo.compare EQ, %[[ARG1]], %[[TMP_366]]
  // CHECK: %[[TMP_368:.*]] = mhlo.and %[[TMP_365]], %[[TMP_367]]
  // CHECK: %[[TMP_369:.*]] = mhlo.constant dense<0x7FC00000>
  // CHECK: %[[TMP_370:.*]] = mhlo.select %[[TMP_368]], %[[TMP_369]], %[[TMP_364]]
  // CHECK: %[[TMP_371:.*]] = mhlo.select %[[TMP_268]], %[[TMP_370]], %[[TMP_266]]
  // CHECK: %[[TMP_372:.*]] = mhlo.floor %[[ARG0]]
  // CHECK: %[[TMP_373:.*]] = mhlo.compare NE, %[[ARG0]], %[[TMP_372]]
  // CHECK: %[[TMP_374:.*]] = mhlo.compare LT, %[[ARG0]], %[[TMP_267]]
  // CHECK: %[[TMP_375:.*]] = mhlo.or %[[TMP_373]], %[[TMP_374]]
  // CHECK: %[[TMP_376:.*]] = mhlo.constant dense<0x7FC00000>
  // CHECK: %[[TMP_377:.*]] = mhlo.select %[[TMP_375]], %[[TMP_376]], %[[TMP_371]]
  %1 = chlo.polygamma %lhs, %rhs : tensor<f32>, tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK: @polygamma_f64
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f64>, %[[ARG1:.*]]: tensor<f64>)
func.func @polygamma_f64(%lhs : tensor<f64>, %rhs : tensor<f64>) -> tensor<f64> {
  // CHECK-DAG: %[[TMP_0:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_1:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_2:.*]] = mhlo.remainder %[[ARG0]], %[[TMP_1]]
  // CHECK: %[[TMP_3:.*]] = mhlo.multiply %[[TMP_1]], %[[TMP_2]]
  // CHECK: %[[TMP_4:.*]] = mhlo.subtract %[[TMP_3]], %[[TMP_0]]
  // CHECK: %[[TMP_5:.*]] = mhlo.add %[[ARG0]], %[[TMP_0]]
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_7:.*]] = mhlo.compare LT, %[[TMP_5]], %[[TMP_6]]
  // CHECK: %[[TMP_8:.*]] = mhlo.negate %[[TMP_5]]
  // CHECK: %[[TMP_9:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_10:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_9]]
  // CHECK: %[[TMP_11:.*]] = mhlo.select %[[TMP_7]], %[[TMP_8]], %[[TMP_10]]
  // CHECK-DAG: %[[TMP_12:.*]] = mhlo.constant dense<0.99999999999980993>
  // CHECK-DAG: %[[TMP_13:.*]] = mhlo.constant dense<676.5203681218851>
  // CHECK-DAG: %[[TMP_14:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_15:.*]] = mhlo.add %[[TMP_11]], %[[TMP_14]]
  // CHECK: %[[TMP_16:.*]] = mhlo.divide %[[TMP_13]], %[[TMP_15]]
  // CHECK: %[[TMP_17:.*]] = mhlo.add %[[TMP_12]], %[[TMP_16]]
  // CHECK-DAG: %[[TMP_18:.*]] = mhlo.constant dense<-1259.1392167224028>
  // CHECK-DAG: %[[TMP_19:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_20:.*]] = mhlo.add %[[TMP_11]], %[[TMP_19]]
  // CHECK: %[[TMP_21:.*]] = mhlo.divide %[[TMP_18]], %[[TMP_20]]
  // CHECK: %[[TMP_22:.*]] = mhlo.add %[[TMP_17]], %[[TMP_21]]
  // CHECK-DAG: %[[TMP_23:.*]] = mhlo.constant dense<771.32342877765313>
  // CHECK-DAG: %[[TMP_24:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_25:.*]] = mhlo.add %[[TMP_11]], %[[TMP_24]]
  // CHECK: %[[TMP_26:.*]] = mhlo.divide %[[TMP_23]], %[[TMP_25]]
  // CHECK: %[[TMP_27:.*]] = mhlo.add %[[TMP_22]], %[[TMP_26]]
  // CHECK-DAG: %[[TMP_28:.*]] = mhlo.constant dense<-176.61502916214059>
  // CHECK-DAG: %[[TMP_29:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_30:.*]] = mhlo.add %[[TMP_11]], %[[TMP_29]]
  // CHECK: %[[TMP_31:.*]] = mhlo.divide %[[TMP_28]], %[[TMP_30]]
  // CHECK: %[[TMP_32:.*]] = mhlo.add %[[TMP_27]], %[[TMP_31]]
  // CHECK-DAG: %[[TMP_33:.*]] = mhlo.constant dense<12.507343278686905>
  // CHECK-DAG: %[[TMP_34:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_35:.*]] = mhlo.add %[[TMP_11]], %[[TMP_34]]
  // CHECK: %[[TMP_36:.*]] = mhlo.divide %[[TMP_33]], %[[TMP_35]]
  // CHECK: %[[TMP_37:.*]] = mhlo.add %[[TMP_32]], %[[TMP_36]]
  // CHECK-DAG: %[[TMP_38:.*]] = mhlo.constant dense<-0.13857109526572012>
  // CHECK-DAG: %[[TMP_39:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_40:.*]] = mhlo.add %[[TMP_11]], %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = mhlo.divide %[[TMP_38]], %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = mhlo.add %[[TMP_37]], %[[TMP_41]]
  // CHECK-DAG: %[[TMP_43:.*]] = mhlo.constant dense<9.9843695780195716E-6>
  // CHECK-DAG: %[[TMP_44:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_45:.*]] = mhlo.add %[[TMP_11]], %[[TMP_44]]
  // CHECK: %[[TMP_46:.*]] = mhlo.divide %[[TMP_43]], %[[TMP_45]]
  // CHECK: %[[TMP_47:.*]] = mhlo.add %[[TMP_42]], %[[TMP_46]]
  // CHECK-DAG: %[[TMP_48:.*]] = mhlo.constant dense<1.5056327351493116E-7>
  // CHECK-DAG: %[[TMP_49:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_50:.*]] = mhlo.add %[[TMP_11]], %[[TMP_49]]
  // CHECK: %[[TMP_51:.*]] = mhlo.divide %[[TMP_48]], %[[TMP_50]]
  // CHECK: %[[TMP_52:.*]] = mhlo.add %[[TMP_47]], %[[TMP_51]]
  // CHECK: %[[TMP_53:.*]] = mhlo.constant dense<7.500000e+00>
  // CHECK: %[[TMP_54:.*]] = mhlo.add %[[TMP_53]], %[[TMP_11]]
  // CHECK: %[[TMP_55:.*]] = mhlo.constant dense<2.0149030205422647>
  // CHECK: %[[TMP_56:.*]] = mhlo.divide %[[TMP_11]], %[[TMP_53]]
  // CHECK: %[[TMP_57:.*]] = mhlo.log_plus_one %[[TMP_56]]
  // CHECK: %[[TMP_58:.*]] = mhlo.add %[[TMP_55]], %[[TMP_57]]
  // CHECK: %[[TMP_59:.*]] = mhlo.divide %[[TMP_54]], %[[TMP_58]]
  // CHECK: %[[TMP_60:.*]] = mhlo.add %[[TMP_11]], %[[TMP_6]]
  // CHECK: %[[TMP_61:.*]] = mhlo.subtract %[[TMP_60]], %[[TMP_59]]
  // CHECK: %[[TMP_62:.*]] = mhlo.multiply %[[TMP_61]], %[[TMP_58]]
  // CHECK: %[[TMP_63:.*]] = mhlo.log %[[TMP_52]]
  // CHECK: %[[TMP_64:.*]] = mhlo.constant dense<0.91893853320467266>
  // CHECK: %[[TMP_65:.*]] = mhlo.add %[[TMP_64]], %[[TMP_62]]
  // CHECK: %[[TMP_66:.*]] = mhlo.add %[[TMP_65]], %[[TMP_63]]
  // CHECK: %[[TMP_67:.*]] = mhlo.abs %[[TMP_5]]
  // CHECK: %[[TMP_68:.*]] = mhlo.floor %[[TMP_67]]
  // CHECK: %[[TMP_69:.*]] = mhlo.subtract %[[TMP_67]], %[[TMP_68]]
  // CHECK: %[[TMP_70:.*]] = mhlo.compare LT, %[[TMP_6]], %[[TMP_69]]
  // CHECK: %[[TMP_71:.*]] = mhlo.subtract %[[TMP_9]], %[[TMP_69]]
  // CHECK: %[[TMP_72:.*]] = mhlo.select %[[TMP_70]], %[[TMP_71]], %[[TMP_69]]
  // CHECK: %[[TMP_73:.*]] = mhlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_74:.*]] = mhlo.multiply %[[TMP_73]], %[[TMP_72]]
  // CHECK: %[[TMP_75:.*]] = mhlo.sine %[[TMP_74]]
  // CHECK: %[[TMP_76:.*]] = mhlo.log %[[TMP_75]]
  // CHECK: %[[TMP_77:.*]] = mhlo.constant dense<1.1447298858494002>
  // CHECK: %[[TMP_78:.*]] = mhlo.subtract %[[TMP_77]], %[[TMP_76]]
  // CHECK: %[[TMP_79:.*]] = mhlo.subtract %[[TMP_78]], %[[TMP_66]]
  // CHECK: %[[TMP_80:.*]] = mhlo.is_finite %[[TMP_76]]
  // CHECK: %[[TMP_81:.*]] = mhlo.negate %[[TMP_76]]
  // CHECK: %[[TMP_82:.*]] = mhlo.select %[[TMP_80]], %[[TMP_79]], %[[TMP_81]]
  // CHECK: %[[TMP_83:.*]] = mhlo.select %[[TMP_7]], %[[TMP_82]], %[[TMP_66]]
  // CHECK: %[[TMP_84:.*]] = mhlo.abs %[[TMP_5]]
  // CHECK: %[[TMP_85:.*]] = mhlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_86:.*]] = mhlo.compare EQ, %[[TMP_84]], %[[TMP_85]]
  // CHECK: %[[TMP_87:.*]] = mhlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_88:.*]] = mhlo.select %[[TMP_86]], %[[TMP_87]], %[[TMP_83]]
  // CHECK: %[[TMP_89:.*]] = mhlo.exponential %[[TMP_88]]
  // CHECK-DAG: %[[TMP_90:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_91:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_92:.*]] = mhlo.negate %[[TMP_5]]
  // CHECK: %[[TMP_93:.*]] = mhlo.power %[[ARG1]], %[[TMP_92]]
  // CHECK: %[[TMP_94:.*]] = mhlo.add %[[ARG1]], %[[TMP_91]]
  // CHECK: %[[TMP_95:.*]] = mhlo.power %[[TMP_94]], %[[TMP_92]]
  // CHECK: %[[TMP_96:.*]] = mhlo.add %[[TMP_93]], %[[TMP_95]]
  // CHECK: %[[TMP_97:.*]] = mhlo.add %[[TMP_94]], %[[TMP_91]]
  // CHECK: %[[TMP_98:.*]] = mhlo.power %[[TMP_97]], %[[TMP_92]]
  // CHECK: %[[TMP_99:.*]] = mhlo.add %[[TMP_96]], %[[TMP_98]]
  // CHECK: %[[TMP_100:.*]] = mhlo.add %[[TMP_97]], %[[TMP_91]]
  // CHECK: %[[TMP_101:.*]] = mhlo.power %[[TMP_100]], %[[TMP_92]]
  // CHECK: %[[TMP_102:.*]] = mhlo.add %[[TMP_99]], %[[TMP_101]]
  // CHECK: %[[TMP_103:.*]] = mhlo.add %[[TMP_100]], %[[TMP_91]]
  // CHECK: %[[TMP_104:.*]] = mhlo.power %[[TMP_103]], %[[TMP_92]]
  // CHECK: %[[TMP_105:.*]] = mhlo.add %[[TMP_102]], %[[TMP_104]]
  // CHECK: %[[TMP_106:.*]] = mhlo.add %[[TMP_103]], %[[TMP_91]]
  // CHECK: %[[TMP_107:.*]] = mhlo.power %[[TMP_106]], %[[TMP_92]]
  // CHECK: %[[TMP_108:.*]] = mhlo.add %[[TMP_105]], %[[TMP_107]]
  // CHECK: %[[TMP_109:.*]] = mhlo.add %[[TMP_106]], %[[TMP_91]]
  // CHECK: %[[TMP_110:.*]] = mhlo.power %[[TMP_109]], %[[TMP_92]]
  // CHECK: %[[TMP_111:.*]] = mhlo.add %[[TMP_108]], %[[TMP_110]]
  // CHECK: %[[TMP_112:.*]] = mhlo.add %[[TMP_109]], %[[TMP_91]]
  // CHECK: %[[TMP_113:.*]] = mhlo.power %[[TMP_112]], %[[TMP_92]]
  // CHECK: %[[TMP_114:.*]] = mhlo.add %[[TMP_111]], %[[TMP_113]]
  // CHECK: %[[TMP_115:.*]] = mhlo.add %[[TMP_112]], %[[TMP_91]]
  // CHECK: %[[TMP_116:.*]] = mhlo.power %[[TMP_115]], %[[TMP_92]]
  // CHECK: %[[TMP_117:.*]] = mhlo.add %[[TMP_114]], %[[TMP_116]]
  // CHECK: %[[TMP_118:.*]] = mhlo.add %[[TMP_115]], %[[TMP_91]]
  // CHECK: %[[TMP_119:.*]] = mhlo.power %[[TMP_118]], %[[TMP_92]]
  // CHECK: %[[TMP_120:.*]] = mhlo.add %[[TMP_117]], %[[TMP_119]]
  // CHECK: %[[TMP_121:.*]] = mhlo.add %[[TMP_118]], %[[TMP_91]]
  // CHECK: %[[TMP_122:.*]] = mhlo.power %[[TMP_121]], %[[TMP_92]]
  // CHECK: %[[TMP_123:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-DAG: %[[TMP_124:.*]] = mhlo.multiply %[[TMP_122]], %[[TMP_121]]
  // CHECK-DAG: %[[TMP_125:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_123]]
  // CHECK: %[[TMP_126:.*]] = mhlo.divide %[[TMP_124]], %[[TMP_125]]
  // CHECK: %[[TMP_127:.*]] = mhlo.multiply %[[TMP_121]], %[[TMP_121]]
  // CHECK: %[[TMP_128:.*]] = mhlo.divide %[[TMP_91]], %[[TMP_127]]
  // CHECK: %[[TMP_129:.*]] = mhlo.constant dense<2.200000e+01>
  // CHECK: %[[TMP_130:.*]] = mhlo.add %[[TMP_5]], %[[TMP_129]]
  // CHECK: %[[TMP_131:.*]] = mhlo.constant dense<2.100000e+01>
  // CHECK: %[[TMP_132:.*]] = mhlo.add %[[TMP_5]], %[[TMP_131]]
  // CHECK: %[[TMP_133:.*]] = mhlo.multiply %[[TMP_130]], %[[TMP_132]]
  // CHECK: %[[TMP_134:.*]] = mhlo.constant dense<-1.3954464685812522E-19>
  // CHECK: %[[TMP_135:.*]] = mhlo.add %[[TMP_90]], %[[TMP_134]]
  // CHECK: %[[TMP_136:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_135]]
  // CHECK: %[[TMP_137:.*]] = mhlo.multiply %[[TMP_133]], %[[TMP_136]]
  // CHECK: %[[TMP_138:.*]] = mhlo.constant dense<2.000000e+01>
  // CHECK: %[[TMP_139:.*]] = mhlo.add %[[TMP_5]], %[[TMP_138]]
  // CHECK: %[[TMP_140:.*]] = mhlo.constant dense<1.900000e+01>
  // CHECK: %[[TMP_141:.*]] = mhlo.add %[[TMP_5]], %[[TMP_140]]
  // CHECK: %[[TMP_142:.*]] = mhlo.multiply %[[TMP_139]], %[[TMP_141]]
  // CHECK: %[[TMP_143:.*]] = mhlo.constant dense<5.5090028283602295E-18>
  // CHECK: %[[TMP_144:.*]] = mhlo.add %[[TMP_137]], %[[TMP_143]]
  // CHECK: %[[TMP_145:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_144]]
  // CHECK: %[[TMP_146:.*]] = mhlo.multiply %[[TMP_142]], %[[TMP_145]]
  // CHECK: %[[TMP_147:.*]] = mhlo.constant dense<1.800000e+01>
  // CHECK: %[[TMP_148:.*]] = mhlo.add %[[TMP_5]], %[[TMP_147]]
  // CHECK: %[[TMP_149:.*]] = mhlo.constant dense<1.700000e+01>
  // CHECK: %[[TMP_150:.*]] = mhlo.add %[[TMP_5]], %[[TMP_149]]
  // CHECK: %[[TMP_151:.*]] = mhlo.multiply %[[TMP_148]], %[[TMP_150]]
  // CHECK: %[[TMP_152:.*]] = mhlo.constant dense<-2.1748686985580617E-16>
  // CHECK: %[[TMP_153:.*]] = mhlo.add %[[TMP_146]], %[[TMP_152]]
  // CHECK: %[[TMP_154:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_153]]
  // CHECK: %[[TMP_155:.*]] = mhlo.multiply %[[TMP_151]], %[[TMP_154]]
  // CHECK: %[[TMP_156:.*]] = mhlo.constant dense<1.600000e+01>
  // CHECK: %[[TMP_157:.*]] = mhlo.add %[[TMP_5]], %[[TMP_156]]
  // CHECK: %[[TMP_158:.*]] = mhlo.constant dense<1.500000e+01>
  // CHECK: %[[TMP_159:.*]] = mhlo.add %[[TMP_5]], %[[TMP_158]]
  // CHECK: %[[TMP_160:.*]] = mhlo.multiply %[[TMP_157]], %[[TMP_159]]
  // CHECK: %[[TMP_161:.*]] = mhlo.constant dense<8.5860620562778452E-15>
  // CHECK: %[[TMP_162:.*]] = mhlo.add %[[TMP_155]], %[[TMP_161]]
  // CHECK: %[[TMP_163:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_162]]
  // CHECK: %[[TMP_164:.*]] = mhlo.multiply %[[TMP_160]], %[[TMP_163]]
  // CHECK: %[[TMP_165:.*]] = mhlo.constant dense<1.400000e+01>
  // CHECK: %[[TMP_166:.*]] = mhlo.add %[[TMP_5]], %[[TMP_165]]
  // CHECK: %[[TMP_167:.*]] = mhlo.constant dense<1.300000e+01>
  // CHECK: %[[TMP_168:.*]] = mhlo.add %[[TMP_5]], %[[TMP_167]]
  // CHECK: %[[TMP_169:.*]] = mhlo.multiply %[[TMP_166]], %[[TMP_168]]
  // CHECK: %[[TMP_170:.*]] = mhlo.constant dense<-3.3896802963225832E-13>
  // CHECK: %[[TMP_171:.*]] = mhlo.add %[[TMP_164]], %[[TMP_170]]
  // CHECK: %[[TMP_172:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_171]]
  // CHECK: %[[TMP_173:.*]] = mhlo.multiply %[[TMP_169]], %[[TMP_172]]
  // CHECK: %[[TMP_174:.*]] = mhlo.constant dense<1.200000e+01>
  // CHECK: %[[TMP_175:.*]] = mhlo.add %[[TMP_5]], %[[TMP_174]]
  // CHECK: %[[TMP_176:.*]] = mhlo.constant dense<1.100000e+01>
  // CHECK: %[[TMP_177:.*]] = mhlo.add %[[TMP_5]], %[[TMP_176]]
  // CHECK: %[[TMP_178:.*]] = mhlo.multiply %[[TMP_175]], %[[TMP_177]]
  // CHECK: %[[TMP_179:.*]] = mhlo.constant dense<1.3382536530684679E-11>
  // CHECK: %[[TMP_180:.*]] = mhlo.add %[[TMP_173]], %[[TMP_179]]
  // CHECK: %[[TMP_181:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_180]]
  // CHECK: %[[TMP_182:.*]] = mhlo.multiply %[[TMP_178]], %[[TMP_181]]
  // CHECK: %[[TMP_183:.*]] = mhlo.constant dense<1.000000e+01>
  // CHECK: %[[TMP_184:.*]] = mhlo.add %[[TMP_5]], %[[TMP_183]]
  // CHECK: %[[TMP_185:.*]] = mhlo.constant dense<9.000000e+00>
  // CHECK: %[[TMP_186:.*]] = mhlo.add %[[TMP_5]], %[[TMP_185]]
  // CHECK: %[[TMP_187:.*]] = mhlo.multiply %[[TMP_184]], %[[TMP_186]]
  // CHECK: %[[TMP_188:.*]] = mhlo.constant dense<-5.2841901386874932E-10>
  // CHECK: %[[TMP_189:.*]] = mhlo.add %[[TMP_182]], %[[TMP_188]]
  // CHECK: %[[TMP_190:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_189]]
  // CHECK: %[[TMP_191:.*]] = mhlo.multiply %[[TMP_187]], %[[TMP_190]]
  // CHECK: %[[TMP_192:.*]] = mhlo.constant dense<8.000000e+00>
  // CHECK: %[[TMP_193:.*]] = mhlo.add %[[TMP_5]], %[[TMP_192]]
  // CHECK: %[[TMP_194:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_195:.*]] = mhlo.add %[[TMP_5]], %[[TMP_194]]
  // CHECK: %[[TMP_196:.*]] = mhlo.multiply %[[TMP_193]], %[[TMP_195]]
  // CHECK: %[[TMP_197:.*]] = mhlo.constant dense<2.08767569878681E-8>
  // CHECK: %[[TMP_198:.*]] = mhlo.add %[[TMP_191]], %[[TMP_197]]
  // CHECK: %[[TMP_199:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_198]]
  // CHECK: %[[TMP_200:.*]] = mhlo.multiply %[[TMP_196]], %[[TMP_199]]
  // CHECK: %[[TMP_201:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_202:.*]] = mhlo.add %[[TMP_5]], %[[TMP_201]]
  // CHECK: %[[TMP_203:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_204:.*]] = mhlo.add %[[TMP_5]], %[[TMP_203]]
  // CHECK: %[[TMP_205:.*]] = mhlo.multiply %[[TMP_202]], %[[TMP_204]]
  // CHECK: %[[TMP_206:.*]] = mhlo.constant dense<-8.2671957671957675E-7>
  // CHECK: %[[TMP_207:.*]] = mhlo.add %[[TMP_200]], %[[TMP_206]]
  // CHECK: %[[TMP_208:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_207]]
  // CHECK: %[[TMP_209:.*]] = mhlo.multiply %[[TMP_205]], %[[TMP_208]]
  // CHECK: %[[TMP_210:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_211:.*]] = mhlo.add %[[TMP_5]], %[[TMP_210]]
  // CHECK: %[[TMP_212:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_213:.*]] = mhlo.add %[[TMP_5]], %[[TMP_212]]
  // CHECK: %[[TMP_214:.*]] = mhlo.multiply %[[TMP_211]], %[[TMP_213]]
  // CHECK: %[[TMP_215:.*]] = mhlo.constant dense<3.3068783068783071E-5>
  // CHECK: %[[TMP_216:.*]] = mhlo.add %[[TMP_209]], %[[TMP_215]]
  // CHECK: %[[TMP_217:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_216]]
  // CHECK: %[[TMP_218:.*]] = mhlo.multiply %[[TMP_214]], %[[TMP_217]]
  // CHECK: %[[TMP_219:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_220:.*]] = mhlo.add %[[TMP_5]], %[[TMP_219]]
  // CHECK: %[[TMP_221:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_222:.*]] = mhlo.add %[[TMP_5]], %[[TMP_221]]
  // CHECK: %[[TMP_223:.*]] = mhlo.multiply %[[TMP_220]], %[[TMP_222]]
  // CHECK: %[[TMP_224:.*]] = mhlo.constant dense<-0.0013888888888888889>
  // CHECK: %[[TMP_225:.*]] = mhlo.add %[[TMP_218]], %[[TMP_224]]
  // CHECK: %[[TMP_226:.*]] = mhlo.multiply %[[TMP_128]], %[[TMP_225]]
  // CHECK: %[[TMP_227:.*]] = mhlo.multiply %[[TMP_223]], %[[TMP_226]]
  // CHECK: %[[TMP_228:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_229:.*]] = mhlo.divide %[[TMP_5]], %[[TMP_121]]
  // CHECK: %[[TMP_230:.*]] = mhlo.constant dense<0.083333333333333329>
  // CHECK: %[[TMP_231:.*]] = mhlo.add %[[TMP_230]], %[[TMP_227]]
  // CHECK: %[[TMP_232:.*]] = mhlo.multiply %[[TMP_229]], %[[TMP_231]]
  // CHECK: %[[TMP_233:.*]] = mhlo.add %[[TMP_228]], %[[TMP_232]]
  // CHECK: %[[TMP_234:.*]] = mhlo.multiply %[[TMP_122]], %[[TMP_233]]
  // CHECK: %[[TMP_235:.*]] = mhlo.add %[[TMP_120]], %[[TMP_126]]
  // CHECK: %[[TMP_236:.*]] = mhlo.add %[[TMP_235]], %[[TMP_234]]
  // CHECK: %[[TMP_237:.*]] = mhlo.abs %[[TMP_122]]
  // CHECK: %[[TMP_238:.*]] = mhlo.abs %[[TMP_120]]
  // CHECK: %[[TMP_239:.*]] = mhlo.constant dense<4.940660e-324>
  // CHECK: %[[TMP_240:.*]] = mhlo.multiply %[[TMP_238]], %[[TMP_239]]
  // CHECK: %[[TMP_241:.*]] = mhlo.compare LT, %[[TMP_237]], %[[TMP_240]]
  // CHECK: %[[TMP_242:.*]] = mhlo.select %[[TMP_241]], %[[TMP_120]], %[[TMP_236]]
  // CHECK: %[[TMP_243:.*]] = mhlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[TMP_244:.*]] = mhlo.compare LT, %[[TMP_5]], %[[TMP_123]]
  // CHECK: %[[TMP_245:.*]] = mhlo.select %[[TMP_244]], %[[TMP_243]], %[[TMP_242]]
  // CHECK: %[[TMP_246:.*]] = mhlo.compare LE, %[[ARG1]], %[[TMP_90]]
  // CHECK: %[[TMP_247:.*]] = mhlo.floor %[[TMP_5]]
  // CHECK: %[[TMP_248:.*]] = mhlo.compare NE, %[[TMP_5]], %[[TMP_247]]
  // CHECK: %[[TMP_249:.*]] = mhlo.and %[[TMP_246]], %[[TMP_248]]
  // CHECK: %[[TMP_250:.*]] = mhlo.select %[[TMP_249]], %[[TMP_243]], %[[TMP_245]]
  // CHECK: %[[TMP_251:.*]] = mhlo.constant dense<0x7FF0000000000000>
  // CHECK: %[[TMP_252:.*]] = mhlo.floor %[[ARG1]]
  // CHECK: %[[TMP_253:.*]] = mhlo.compare EQ, %[[ARG1]], %[[TMP_252]]
  // CHECK: %[[TMP_254:.*]] = mhlo.and %[[TMP_246]], %[[TMP_253]]
  // CHECK: %[[TMP_255:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_256:.*]] = mhlo.floor %[[TMP_5]]
  // CHECK: %[[TMP_257:.*]] = mhlo.compare EQ, %[[TMP_5]], %[[TMP_256]]
  // CHECK: %[[TMP_258:.*]] = mhlo.remainder %[[TMP_5]], %[[TMP_255]]
  // CHECK: %[[TMP_259:.*]] = mhlo.compare EQ, %[[TMP_258]], %[[TMP_90]]
  // CHECK: %[[TMP_260:.*]] = mhlo.and %[[TMP_257]], %[[TMP_259]]
  // CHECK: %[[TMP_261:.*]] = mhlo.select %[[TMP_260]], %[[TMP_251]], %[[TMP_243]]
  // CHECK: %[[TMP_262:.*]] = mhlo.select %[[TMP_254]], %[[TMP_261]], %[[TMP_250]]
  // CHECK: %[[TMP_263:.*]] = mhlo.compare EQ, %[[TMP_5]], %[[TMP_91]]
  // CHECK: %[[TMP_264:.*]] = mhlo.select %[[TMP_263]], %[[TMP_251]], %[[TMP_262]]
  // CHECK: %[[TMP_265:.*]] = mhlo.multiply %[[TMP_4]], %[[TMP_89]]
  // CHECK: %[[TMP_266:.*]] = mhlo.multiply %[[TMP_265]], %[[TMP_264]]
  // CHECK: %[[TMP_267:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_268:.*]] = mhlo.compare EQ, %[[ARG0]], %[[TMP_267]]
  // CHECK: %[[TMP_269:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_270:.*]] = mhlo.compare LT, %[[ARG1]], %[[TMP_269]]
  // CHECK: %[[TMP_271:.*]] = mhlo.negate %[[ARG1]]
  // CHECK: %[[TMP_272:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_273:.*]] = mhlo.subtract %[[ARG1]], %[[TMP_272]]
  // CHECK: %[[TMP_274:.*]] = mhlo.select %[[TMP_270]], %[[TMP_271]], %[[TMP_273]]
  // CHECK-DAG: %[[TMP_275:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TMP_276:.*]] = mhlo.constant dense<0.99999999999980993>
  // CHECK-DAG: %[[TMP_277:.*]] = mhlo.constant dense<676.5203681218851>
  // CHECK-DAG: %[[TMP_278:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_279:.*]] = mhlo.add %[[TMP_274]], %[[TMP_278]]
  // CHECK: %[[TMP_280:.*]] = mhlo.multiply %[[TMP_279]], %[[TMP_279]]
  // CHECK: %[[TMP_281:.*]] = mhlo.divide %[[TMP_277]], %[[TMP_280]]
  // CHECK: %[[TMP_282:.*]] = mhlo.subtract %[[TMP_275]], %[[TMP_281]]
  // CHECK: %[[TMP_283:.*]] = mhlo.divide %[[TMP_277]], %[[TMP_279]]
  // CHECK: %[[TMP_284:.*]] = mhlo.add %[[TMP_276]], %[[TMP_283]]
  // CHECK-DAG: %[[TMP_285:.*]] = mhlo.constant dense<-1259.1392167224028>
  // CHECK-DAG: %[[TMP_286:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_287:.*]] = mhlo.add %[[TMP_274]], %[[TMP_286]]
  // CHECK: %[[TMP_288:.*]] = mhlo.multiply %[[TMP_287]], %[[TMP_287]]
  // CHECK: %[[TMP_289:.*]] = mhlo.divide %[[TMP_285]], %[[TMP_288]]
  // CHECK: %[[TMP_290:.*]] = mhlo.subtract %[[TMP_282]], %[[TMP_289]]
  // CHECK: %[[TMP_291:.*]] = mhlo.divide %[[TMP_285]], %[[TMP_287]]
  // CHECK: %[[TMP_292:.*]] = mhlo.add %[[TMP_284]], %[[TMP_291]]
  // CHECK-DAG: %[[TMP_293:.*]] = mhlo.constant dense<771.32342877765313>
  // CHECK-DAG: %[[TMP_294:.*]] = mhlo.constant dense<3.000000e+00>
  // CHECK: %[[TMP_295:.*]] = mhlo.add %[[TMP_274]], %[[TMP_294]]
  // CHECK: %[[TMP_296:.*]] = mhlo.multiply %[[TMP_295]], %[[TMP_295]]
  // CHECK: %[[TMP_297:.*]] = mhlo.divide %[[TMP_293]], %[[TMP_296]]
  // CHECK: %[[TMP_298:.*]] = mhlo.subtract %[[TMP_290]], %[[TMP_297]]
  // CHECK: %[[TMP_299:.*]] = mhlo.divide %[[TMP_293]], %[[TMP_295]]
  // CHECK: %[[TMP_300:.*]] = mhlo.add %[[TMP_292]], %[[TMP_299]]
  // CHECK-DAG: %[[TMP_301:.*]] = mhlo.constant dense<-176.61502916214059>
  // CHECK-DAG: %[[TMP_302:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_303:.*]] = mhlo.add %[[TMP_274]], %[[TMP_302]]
  // CHECK: %[[TMP_304:.*]] = mhlo.multiply %[[TMP_303]], %[[TMP_303]]
  // CHECK: %[[TMP_305:.*]] = mhlo.divide %[[TMP_301]], %[[TMP_304]]
  // CHECK: %[[TMP_306:.*]] = mhlo.subtract %[[TMP_298]], %[[TMP_305]]
  // CHECK: %[[TMP_307:.*]] = mhlo.divide %[[TMP_301]], %[[TMP_303]]
  // CHECK: %[[TMP_308:.*]] = mhlo.add %[[TMP_300]], %[[TMP_307]]
  // CHECK-DAG: %[[TMP_309:.*]] = mhlo.constant dense<12.507343278686905>
  // CHECK-DAG: %[[TMP_310:.*]] = mhlo.constant dense<5.000000e+00>
  // CHECK: %[[TMP_311:.*]] = mhlo.add %[[TMP_274]], %[[TMP_310]]
  // CHECK: %[[TMP_312:.*]] = mhlo.multiply %[[TMP_311]], %[[TMP_311]]
  // CHECK: %[[TMP_313:.*]] = mhlo.divide %[[TMP_309]], %[[TMP_312]]
  // CHECK: %[[TMP_314:.*]] = mhlo.subtract %[[TMP_306]], %[[TMP_313]]
  // CHECK: %[[TMP_315:.*]] = mhlo.divide %[[TMP_309]], %[[TMP_311]]
  // CHECK: %[[TMP_316:.*]] = mhlo.add %[[TMP_308]], %[[TMP_315]]
  // CHECK-DAG: %[[TMP_317:.*]] = mhlo.constant dense<-0.13857109526572012>
  // CHECK-DAG: %[[TMP_318:.*]] = mhlo.constant dense<6.000000e+00>
  // CHECK: %[[TMP_319:.*]] = mhlo.add %[[TMP_274]], %[[TMP_318]]
  // CHECK: %[[TMP_320:.*]] = mhlo.multiply %[[TMP_319]], %[[TMP_319]]
  // CHECK: %[[TMP_321:.*]] = mhlo.divide %[[TMP_317]], %[[TMP_320]]
  // CHECK: %[[TMP_322:.*]] = mhlo.subtract %[[TMP_314]], %[[TMP_321]]
  // CHECK: %[[TMP_323:.*]] = mhlo.divide %[[TMP_317]], %[[TMP_319]]
  // CHECK: %[[TMP_324:.*]] = mhlo.add %[[TMP_316]], %[[TMP_323]]
  // CHECK-DAG: %[[TMP_325:.*]] = mhlo.constant dense<9.9843695780195716E-6>
  // CHECK-DAG: %[[TMP_326:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_327:.*]] = mhlo.add %[[TMP_274]], %[[TMP_326]]
  // CHECK: %[[TMP_328:.*]] = mhlo.multiply %[[TMP_327]], %[[TMP_327]]
  // CHECK: %[[TMP_329:.*]] = mhlo.divide %[[TMP_325]], %[[TMP_328]]
  // CHECK: %[[TMP_330:.*]] = mhlo.subtract %[[TMP_322]], %[[TMP_329]]
  // CHECK: %[[TMP_331:.*]] = mhlo.divide %[[TMP_325]], %[[TMP_327]]
  // CHECK: %[[TMP_332:.*]] = mhlo.add %[[TMP_324]], %[[TMP_331]]
  // CHECK-DAG: %[[TMP_333:.*]] = mhlo.constant dense<1.5056327351493116E-7>
  // CHECK-DAG: %[[TMP_334:.*]] = mhlo.constant dense<8.000000e+00>
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
  // CHECK: %[[TMP_345:.*]] = mhlo.log_plus_one %[[TMP_344]]
  // CHECK: %[[TMP_346:.*]] = mhlo.add %[[TMP_343]], %[[TMP_345]]
  // CHECK: %[[TMP_347:.*]] = mhlo.divide %[[TMP_338]], %[[TMP_340]]
  // CHECK: %[[TMP_348:.*]] = mhlo.constant dense<7.000000e+00>
  // CHECK: %[[TMP_349:.*]] = mhlo.divide %[[TMP_348]], %[[TMP_342]]
  // CHECK: %[[TMP_350:.*]] = mhlo.add %[[TMP_346]], %[[TMP_347]]
  // CHECK: %[[TMP_351:.*]] = mhlo.subtract %[[TMP_350]], %[[TMP_349]]
  // CHECK: %[[TMP_352:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK: %[[TMP_353:.*]] = mhlo.add %[[ARG1]], %[[TMP_352]]
  // CHECK: %[[TMP_354:.*]] = mhlo.floor %[[TMP_353]]
  // CHECK: %[[TMP_355:.*]] = mhlo.abs %[[TMP_354]]
  // CHECK: %[[TMP_356:.*]] = mhlo.add %[[ARG1]], %[[TMP_355]]
  // CHECK: %[[TMP_357:.*]] = mhlo.constant dense<3.1415926535897931>
  // CHECK: %[[TMP_358:.*]] = mhlo.multiply %[[TMP_357]], %[[TMP_356]]
  // CHECK: %[[TMP_359:.*]] = mhlo.cosine %[[TMP_358]]
  // CHECK: %[[TMP_360:.*]] = mhlo.sine %[[TMP_358]]
  // CHECK: %[[TMP_361:.*]] = mhlo.multiply %[[TMP_357]], %[[TMP_359]]
  // CHECK: %[[TMP_362:.*]] = mhlo.divide %[[TMP_361]], %[[TMP_360]]
  // CHECK: %[[TMP_363:.*]] = mhlo.subtract %[[TMP_351]], %[[TMP_362]]
  // CHECK: %[[TMP_364:.*]] = mhlo.select %[[TMP_270]], %[[TMP_363]], %[[TMP_351]]
  // CHECK: %[[TMP_365:.*]] = mhlo.compare LE, %[[ARG1]], %[[TMP_275]]
  // CHECK: %[[TMP_366:.*]] = mhlo.floor %[[ARG1]]
  // CHECK: %[[TMP_367:.*]] = mhlo.compare EQ, %[[ARG1]], %[[TMP_366]]
  // CHECK: %[[TMP_368:.*]] = mhlo.and %[[TMP_365]], %[[TMP_367]]
  // CHECK: %[[TMP_369:.*]] = mhlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[TMP_370:.*]] = mhlo.select %[[TMP_368]], %[[TMP_369]], %[[TMP_364]]
  // CHECK: %[[TMP_371:.*]] = mhlo.select %[[TMP_268]], %[[TMP_370]], %[[TMP_266]]
  // CHECK: %[[TMP_372:.*]] = mhlo.floor %[[ARG0]]
  // CHECK: %[[TMP_373:.*]] = mhlo.compare NE, %[[ARG0]], %[[TMP_372]]
  // CHECK: %[[TMP_374:.*]] = mhlo.compare LT, %[[ARG0]], %[[TMP_267]]
  // CHECK: %[[TMP_375:.*]] = mhlo.or %[[TMP_373]], %[[TMP_374]]
  // CHECK: %[[TMP_376:.*]] = mhlo.constant dense<0x7FF8000000000000>
  // CHECK: %[[TMP_377:.*]] = mhlo.select %[[TMP_375]], %[[TMP_376]], %[[TMP_371]]
  %1 = chlo.polygamma %lhs, %rhs : tensor<f64>, tensor<f64> -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @polygamma_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>, %[[ARG1:.*]]: tensor<f16>)
func.func @polygamma_f16(%lhs : tensor<f16>, %rhs : tensor<f16>) -> tensor<f16> {
  // CHECK: mhlo.convert %[[ARG0]] : (tensor<f16>) -> tensor<f32>
  // CHECK: mhlo.convert %[[ARG1]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = mhlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.polygamma %lhs, %rhs : tensor<f16>, tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----


// CHECK-LABEL: @sinh_f32
// CHECK-SAME: (%[[X:.*]]: tensor<f32>)
func.func @sinh_f32(%x : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[HALF:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[LOG_HALF:.*]] = mhlo.log %[[HALF]] : tensor<f32>
  // CHECK: %[[X_PLUS_LOG_HALF:.*]] = mhlo.add %[[X]], %[[LOG_HALF]] : tensor<f32>
  // CHECK: %[[EXP_1:.*]] = mhlo.exponential %[[X_PLUS_LOG_HALF]] : tensor<f32>
  // CHECK: %[[LOG_HALF_MINUS_X:.*]] = mhlo.subtract %[[LOG_HALF]], %[[X]] : tensor<f32>
  // CHECK: %[[EXP_2:.*]] = mhlo.exponential %[[LOG_HALF_MINUS_X]] : tensor<f32>
  // CHECK: %[[LARGE_SINH_RESULT:.*]] = mhlo.subtract %[[EXP_1]], %[[EXP_2]] : tensor<f32>
  // CHECK: %[[EXPM1:.*]] = mhlo.exponential_minus_one %[[X]] : tensor<f32>
  // CHECK-DAG: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[HALF:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[EXPM1_PLUS_ONE:.*]] = mhlo.add %[[EXPM1]], %[[ONE]] : tensor<f32>
  // CHECK: %[[RATIO:.*]] = mhlo.divide %[[EXPM1]], %[[EXPM1_PLUS_ONE]] : tensor<f32>
  // CHECK: %[[SUM:.*]] = mhlo.add %[[EXPM1]], %[[RATIO]] : tensor<f32>
  // CHECK: %[[SMALL_SINH_RESULT:.*]] = mhlo.multiply %[[HALF]], %[[SUM]] : tensor<f32>
  // CHECK: %[[ABS_X:.*]] = mhlo.abs %[[X]] : tensor<f32>
  // CHECK: %[[ABS_X_LT_ONE:.*]] = mhlo.compare LT, %[[ABS_X]], %[[ONE]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: %[[RESULT:.*]] = mhlo.select %[[ABS_X_LT_ONE]], %[[SMALL_SINH_RESULT]], %[[LARGE_SINH_RESULT]] : tensor<i1>, tensor<f32>
  // CHECK: return %[[RESULT]] : tensor<f32>
  %1 = chlo.sinh %x : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @sinh_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>)
func.func @sinh_f16(%x : tensor<f16>) -> tensor<f16> {
  // CHECK: mhlo.convert %[[ARG0]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = mhlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.sinh %x : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @sinh_complex
// CHECK-SAME: (%[[X:.*]]: tensor<2xcomplex<f32>>)
func.func @sinh_complex(%x : tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK: %[[HALF:.*]] = mhlo.constant dense<(5.000000e-01,0.000000e+00)> : tensor<2xcomplex<f32>>
  // CHECK: %[[LOG_HALF:.*]] = mhlo.log %[[HALF]] : tensor<2xcomplex<f32>>
  // CHECK: %[[X_PLUS_LOG_HALF:.*]] = mhlo.add %[[X]], %[[LOG_HALF]] : tensor<2xcomplex<f32>>
  // CHECK: %[[EXP_1:.*]] = mhlo.exponential %[[X_PLUS_LOG_HALF]] : tensor<2xcomplex<f32>>
  // CHECK: %[[LOG_HALF_MINUS_X:.*]] = mhlo.subtract %[[LOG_HALF]], %[[X]] : tensor<2xcomplex<f32>>
  // CHECK: %[[EXP_2:.*]] = mhlo.exponential %[[LOG_HALF_MINUS_X]] : tensor<2xcomplex<f32>>
  // CHECK: %[[RESULT:.*]] = mhlo.subtract %[[EXP_1]], %[[EXP_2]] : tensor<2xcomplex<f32>>
  // CHECK: return %[[RESULT]] : tensor<2xcomplex<f32>>
  %1 = chlo.sinh %x : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
  func.return %1 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: @cosh_f32
// CHECK-SAME: (%[[X:.*]]: tensor<f32>)
func.func @cosh_f32(%x : tensor<f32>) -> tensor<f32> {
  // CHECK: %[[HALF:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[LOG_HALF:.*]] = mhlo.log %[[HALF]] : tensor<f32>
  // CHECK: %[[X_PLUS_LOG_HALF:.*]] = mhlo.add %[[X]], %[[LOG_HALF]] : tensor<f32>
  // CHECK: %[[EXP_1:.*]] = mhlo.exponential %[[X_PLUS_LOG_HALF]] : tensor<f32>
  // CHECK: %[[LOG_HALF_MINUS_X:.*]] = mhlo.subtract %[[LOG_HALF]], %[[X]] : tensor<f32>
  // CHECK: %[[EXP_2:.*]] = mhlo.exponential %[[LOG_HALF_MINUS_X]] : tensor<f32>
  // CHECK: %[[RESULT:.*]] = mhlo.add %[[EXP_1]], %[[EXP_2]] : tensor<f32>
  // CHECK: return %[[RESULT]] : tensor<f32>
  %1 = chlo.cosh %x : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @cosh_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>)
func.func @cosh_f16(%x : tensor<f16>) -> tensor<f16> {
  // CHECK: mhlo.convert %[[ARG0]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = mhlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RES]]
  %1 = chlo.cosh %x : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @cosh_complex_f32
// CHECK-SAME: (%[[X:.*]]: tensor<complex<f32>>)
func.func @cosh_complex_f32(%x : tensor<complex<f32>>) -> tensor<complex<f32>> {
  // CHECK: %[[HALF:.*]] = mhlo.constant dense<(5.000000e-01,0.000000e+00)> : tensor<complex<f32>>
  // CHECK: %[[LOG_HALF:.*]] = mhlo.log %[[HALF]] : tensor<complex<f32>>
  // CHECK: %[[X_PLUS_LOG_HALF:.*]] = mhlo.add %[[X]], %[[LOG_HALF]] : tensor<complex<f32>>
  // CHECK: %[[EXP_1:.*]] = mhlo.exponential %[[X_PLUS_LOG_HALF]] : tensor<complex<f32>>
  // CHECK: %[[LOG_HALF_MINUS_X:.*]] = mhlo.subtract %[[LOG_HALF]], %[[X]] : tensor<complex<f32>>
  // CHECK: %[[EXP_2:.*]] = mhlo.exponential %[[LOG_HALF_MINUS_X]] : tensor<complex<f32>>
  // CHECK: %[[RESULT:.*]] = mhlo.add %[[EXP_1]], %[[EXP_2]] : tensor<complex<f32>>
  // CHECK: return %[[RESULT]] : tensor<complex<f32>>
  %1 = chlo.cosh %x : tensor<complex<f32>> -> tensor<complex<f32>>
  func.return %1 : tensor<complex<f32>>
}

// -----

// CHECK-LABEL: @atanh_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @atanh_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %[[TMP_0:.*]] = mhlo.abs %[[ARG]]
  // CHECK-NEXT: %[[TMP_1:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[TMP_2:.*]] = mhlo.compare GT, %[[TMP_0]], %[[TMP_1]]
  // CHECK-NEXT: %[[TMP_3:.*]] = mhlo.constant dense<0x7FC00000>
  // CHECK-NEXT: %[[TMP_4:.*]] = mhlo.log_plus_one %[[ARG]]
  // CHECK-NEXT: %[[TMP_5:.*]] = mhlo.negate %[[ARG]]
  // CHECK-NEXT: %[[TMP_6:.*]] = mhlo.log_plus_one %[[TMP_5]]
  // CHECK-NEXT: %[[TMP_7:.*]] = mhlo.subtract %[[TMP_4]], %[[TMP_6]]
  // CHECK-NEXT: %[[TMP_8:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK-NEXT: %[[TMP_9:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_8]]
  // CHECK-NEXT: %[[TMP_10:.*]] = mhlo.select %[[TMP_2]], %[[TMP_3]], %[[TMP_9]]
  // CHECK-NEXT: return %[[TMP_10]]
  %result = "chlo.atanh"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// -----

// CHECK-LABEL: @atanh_complex_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<complex<f32>>
func.func @atanh_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  // CHECK-NEXT: %[[REAL:.*]] = mhlo.real %[[ARG]]
  // CHECK-NEXT: %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[CMP0:.*]] = mhlo.compare  GE, %[[REAL]], %[[ZERO]]
  // CHECK-NEXT: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[NEG_ONE:.*]] = mhlo.constant dense<-1.000000e+00>
  // CHECK-NEXT: %[[SELECT0:.*]] = mhlo.select %[[CMP0]], %[[ONE]], %[[NEG_ONE]]
  // CHECK-NEXT: %[[FOUR:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK-NEXT: %[[ABS_REAL:.*]] = mhlo.abs %[[REAL]]
  // CHECK-NEXT: %[[LARGE_VAL:.*]] = mhlo.constant dense<3.40282347E+38>
  // CHECK-NEXT: %[[INF:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK-NEXT: %[[CMP1:.*]] = mhlo.compare  GT, %[[LARGE_VAL]], %[[INF]]
  // CHECK-NEXT: %[[LARGE_VAL2:.*]] = mhlo.constant dense<9.00719925E+15>
  // CHECK-NEXT: %[[LARGE_VAL3:.*]] = mhlo.constant dense<9.99999968E+37>
  // CHECK-NEXT: %[[CMP2:.*]] = mhlo.compare  GT, %[[LARGE_VAL]], %[[LARGE_VAL3]]
  // CHECK-NEXT: %[[LARGE_VAL4:.*]] = mhlo.constant dense<0x4B800001>
  // CHECK-NEXT: %[[LARGE_VAL5:.*]] = mhlo.constant dense<2.050000e+03>
  // CHECK-NEXT: %[[SELECT1:.*]] = mhlo.select %[[CMP2]], %[[LARGE_VAL4]], %[[LARGE_VAL5]]
  // CHECK-NEXT: %[[SELECT2:.*]] = mhlo.select %[[CMP1]], %[[LARGE_VAL2]], %[[SELECT1]]
  // CHECK-NEXT: %[[SQUARE:.*]] = mhlo.multiply %[[SELECT2]], %[[SELECT2]]
  // CHECK-NEXT: %[[CMP3:.*]] = mhlo.compare  LT, %[[ABS_REAL]], %[[SQUARE]]
  // CHECK-NEXT: %[[IMAG:.*]] = mhlo.imag %[[ARG]]
  // CHECK-NEXT: %[[ABS_IMAG:.*]] = mhlo.abs %[[IMAG]]
  // CHECK-NEXT: %[[CMP4:.*]] = mhlo.compare LT, %[[ABS_IMAG]], %[[SQUARE]]
  // CHECK-NEXT: %[[AND:.*]] = mhlo.and %[[CMP3]], %[[CMP4]]
  // CHECK-NEXT: %[[SUB0:.*]] = mhlo.subtract %[[ONE]], %[[ABS_REAL]]
  // CHECK-NEXT: %[[SQUARE1:.*]] = mhlo.multiply %[[SUB0]], %[[SUB0]]
  // CHECK-NEXT: %[[SQUARE2:.*]] = mhlo.multiply %[[IMAG]], %[[IMAG]]
  // CHECK-NEXT: %[[ADD0:.*]] = mhlo.add %[[SQUARE1]], %[[SQUARE2]]
  // CHECK-NEXT: %[[DIV0:.*]] = mhlo.divide %[[ABS_REAL]], %[[ADD0]]
  // CHECK-NEXT: %[[MULT0:.*]] = mhlo.multiply %[[ABS_IMAG]], %[[SELECT2]]
  // CHECK-NEXT: %[[CMP5:.*]] = mhlo.compare LT, %[[MULT0]], %[[ABS_REAL]]
  // CHECK-NEXT: %[[DIV1:.*]] = mhlo.divide %[[ONE]], %[[ABS_REAL]]
  // CHECK-NEXT: %[[ISINF_REAL:.*]] = mhlo.constant dense<0x7F800000>
  // CHECK-NEXT: %[[ISNINF_REAL:.*]] = mhlo.compare  EQ, %[[REAL]], %[[ISINF_REAL]]
  // CHECK-NEXT: %[[ISINF_REAL_1:.*]] = mhlo.constant dense<0xFF800000>
  // CHECK-NEXT: %[[ISNINF_REAL_1:.*]] = mhlo.compare  EQ, %[[REAL]], %[[ISINF_REAL_1]]
  // CHECK-NEXT: %[[OR0:.*]] = mhlo.or %[[ISNINF_REAL]], %[[ISNINF_REAL_1]]
  // CHECK-NEXT: %[[ISINF_IMAG:.*]] = mhlo.compare  EQ, %[[IMAG]], %[[ISINF_REAL]]
  // CHECK-NEXT: %[[ISNINF_IMAG:.*]] = mhlo.compare  EQ, %[[IMAG]], %[[ISINF_REAL_1]]
  // CHECK-NEXT: %[[OR1:.*]] = mhlo.or %[[ISINF_IMAG]], %[[ISNINF_IMAG]]
  // CHECK-NEXT: %[[OR2:.*]] = mhlo.or %[[OR0]], %[[OR1]]
  // CHECK-NEXT: %[[DIV2:.*]] = mhlo.divide %[[ABS_REAL]], %[[IMAG]]
  // CHECK-NEXT: %[[DIV3:.*]] = mhlo.divide %[[IMAG]], %[[ABS_REAL]]
  // CHECK-NEXT: %[[ADD1:.*]] = mhlo.add %[[DIV2]], %[[DIV3]]
  // CHECK-NEXT: %[[DIV4:.*]] = mhlo.divide %[[ONE]], %[[ADD1]]
  // CHECK-NEXT: %[[DIV5:.*]] = mhlo.divide %[[DIV4]], %[[IMAG]]
  // CHECK-NEXT: %[[SELECT3:.*]] = mhlo.select %[[OR2]], %[[ZERO]], %[[DIV5]]
  // CHECK-NEXT: %[[SELECT4:.*]] = mhlo.select %[[CMP5]], %[[DIV1]], %[[SELECT3]]
  // CHECK-NEXT: %[[SELECT5:.*]] = mhlo.select %[[AND]], %[[DIV0]], %[[SELECT4]]
  // CHECK-NEXT: %[[MULT1:.*]] = mhlo.multiply %[[FOUR]], %[[SELECT5]]
  // CHECK-NEXT: %[[LOG1P:.*]] = mhlo.log_plus_one %[[MULT1]]
  // CHECK-NEXT: %[[MULT2:.*]] = mhlo.multiply %[[SELECT0]], %[[LOG1P]]
  // CHECK-NEXT: %[[HALF:.*]] = mhlo.constant dense<2.500000e-01>
  // CHECK-NEXT: %[[MULT3:.*]] = mhlo.multiply %[[MULT2]], %[[HALF]]
  // CHECK-NEXT: %[[ADD2:.*]] = mhlo.add %[[IMAG]], %[[IMAG]]
  // CHECK-NEXT: %[[ADD3:.*]] = mhlo.add %[[ONE]], %[[ABS_REAL]]
  // CHECK-NEXT: %[[MULT4:.*]] = mhlo.multiply %[[SUB0]], %[[ADD3]]
  // CHECK-NEXT: %[[SUB1:.*]] = mhlo.subtract %[[MULT4]], %[[SQUARE2]]
  // CHECK-NEXT: %[[ATAN2:.*]] = mhlo.atan2 %[[ADD2]], %[[SUB1]]
  // CHECK-NEXT: %[[CONST0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[CMP6:.*]] = mhlo.compare  GE, %[[IMAG]], %[[CONST0]]
  // CHECK-NEXT: %[[SELECT6:.*]] = mhlo.select %[[CMP6]], %[[ONE]], %[[NEG_ONE]]
  // CHECK-NEXT: %[[PI:.*]] = mhlo.constant dense<3.14159274>
  // CHECK-NEXT: %[[MULT5:.*]] = mhlo.multiply %[[SELECT6]], %[[PI]]
  // CHECK-NEXT: %[[SELECT7:.*]] = mhlo.select %[[AND]], %[[ATAN2]], %[[MULT5]]
  // CHECK-NEXT: %[[CONST:.*]] = mhlo.constant dense<5.000000e-01>
  // CHECK-NEXT: %[[MULT6:.*]] = mhlo.multiply %[[SELECT7]], %[[CONST]]
  // CHECK-NEXT: %[[FINAL_RESULT:.*]] = mhlo.complex %[[MULT3]], %[[MULT6]]
  // CHECK-NEXT: return %[[FINAL_RESULT]] : tensor<complex<f32>>
  %result = "chlo.atanh"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// CHECK-LABEL: @next_after_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xf32>, %[[ARG1:.*]]: tensor<2xf32>)
func.func @next_after_f32(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: %[[X_AS_INT:.*]] = mhlo.bitcast_convert %[[ARG0]] : (tensor<2xf32>) -> tensor<2xi32>
  // CHECK: %[[Y_AS_INT:.*]] = mhlo.bitcast_convert %[[ARG1]] : (tensor<2xf32>) -> tensor<2xi32>
  // CHECK: %[[X_IS_NAN:.*]] = mhlo.compare NE, %[[ARG0]], %[[ARG0]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHECK: %[[Y_IS_NAN:.*]] = mhlo.compare NE, %[[ARG1]], %[[ARG1]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHECK: %[[INPUT_IS_NAN:.*]] = mhlo.or %[[X_IS_NAN]], %[[Y_IS_NAN]] : tensor<2xi1>
  // CHECK: %[[NAN:.*]] = mhlo.constant dense<0x7FC00000> : tensor<2xf32>
  // CHECK: %[[NAN_AS_INT:.*]] = mhlo.bitcast_convert %[[NAN]] : (tensor<2xf32>) -> tensor<2xi32>
  // CHECK-DAG: %[[SIGN_MASK:.*]] = mhlo.constant dense<-2147483648> : tensor<2xi32>
  // CHECK-DAG: %[[NEGATED_SIGN_MASK:.*]] = mhlo.constant dense<2147483647> : tensor<2xi32>
  // CHECK: %[[X_ABS:.*]] = mhlo.and %[[X_AS_INT]], %[[NEGATED_SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[Y_ABS:.*]] = mhlo.and %[[Y_AS_INT]], %[[NEGATED_SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[X_AND_Y_ARE_EQUAL:.*]] = mhlo.compare EQ, %[[ARG0]], %[[ARG1]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHECK: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<2xi32>
  // CHECK: %[[X_ABS_IS_ZERO:.*]] = mhlo.compare EQ, %[[X_ABS]], %[[ZERO]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[Y_ABS_IS_ZERO:.*]] = mhlo.compare EQ, %[[Y_ABS]], %[[ZERO]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[X_SIGN:.*]] = mhlo.and %[[X_AS_INT]], %[[SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[Y_SIGN:.*]] = mhlo.and %[[Y_AS_INT]], %[[SIGN_MASK]] : tensor<2xi32>
  // CHECK: %[[ONE:.*]] = mhlo.constant dense<1> : tensor<2xi32>
  // CHECK: %[[RESULT_FOR_X_ZERO_Y_NON_ZERO:.*]] = mhlo.or %[[Y_SIGN]], %[[ONE]] : tensor<2xi32>
  // CHECK: %[[SIGNS_DISAGREE:.*]] = mhlo.compare NE, %[[X_SIGN]], %[[Y_SIGN]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[X_MAGNITUDE_LARGER_THAN_Y:.*]] = mhlo.compare GT, %[[X_ABS]], %[[Y_ABS]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: %[[RESULT_HAS_SMALLER_MAGNITUDE:.*]] = mhlo.or %[[X_MAGNITUDE_LARGER_THAN_Y]], %[[SIGNS_DISAGREE]] : tensor<2xi1>
  // CHECK: %[[MINUS_ONE:.*]] = mhlo.constant dense<-1> : tensor<2xi32>
  // CHECK: %[[MAGNITUDE_ADJUSTMENT:.*]] = mhlo.select %[[RESULT_HAS_SMALLER_MAGNITUDE]], %[[MINUS_ONE]], %[[ONE]] : tensor<2xi1>, tensor<2xi32>
  // CHECK: %[[RESULT0:.*]] = mhlo.add %[[X_AS_INT]], %[[MAGNITUDE_ADJUSTMENT]] : tensor<2xi32>
  // CHECK: %[[RESULT1:.*]] = mhlo.select %[[Y_ABS_IS_ZERO]], %[[Y_AS_INT]], %[[RESULT_FOR_X_ZERO_Y_NON_ZERO]] : tensor<2xi1>, tensor<2xi32>
  // CHECK: %[[RESULT2:.*]] = mhlo.select %[[X_ABS_IS_ZERO]], %[[RESULT1]], %[[RESULT0]] : tensor<2xi1>, tensor<2xi32>
  // CHECK: %[[RESULT3:.*]] = mhlo.select %[[X_AND_Y_ARE_EQUAL]], %[[Y_AS_INT]], %[[RESULT2]] : tensor<2xi1>, tensor<2xi32>
  // CHECK: %[[RESULT4:.*]] = mhlo.select %[[INPUT_IS_NAN]], %[[NAN_AS_INT]], %[[RESULT3]] : tensor<2xi1>, tensor<2xi32>
  // CHECK: %[[FINAL_RESULT:.*]] = mhlo.bitcast_convert %[[RESULT4]] : (tensor<2xi32>) -> tensor<2xf32>
  // CHECK: return %[[FINAL_RESULT]]
  %1 = chlo.broadcast_next_after %x, %y : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %1 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @top_k
// CHECK-SAME: (%[[ARG:.*]]: tensor<16x16xf32>)
func.func @top_k(%arg : tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>) {
  // CHECK-HIGH-LEVEL: mhlo.topk
  // CHECK: %values, %indices = mhlo.topk(%arg0, k = 8) : tensor<16x16xf32> -> (tensor<16x8xf32>, tensor<16x8xi32>)
  %1:2 = chlo.top_k(%arg, k=8) : tensor<16x16xf32> -> (tensor<16x8xf32>, tensor<16x8xi32>)
  func.return %1#0, %1#1 : tensor<16x8xf32>, tensor<16x8xi32>
}

// -----

// CHECK-LABEL: @dyn_top_k
// CHECK-SAME: ([[ARG:%.*]]: tensor<?x5x?xi1>
// CHECK-SAME: -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>)
func.func @dyn_top_k(%arg0: tensor<?x5x?xi1>) -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>) {
  // CHECK-HIGH-LEVEL: mhlo.topk
  // CHECK: %values, %indices = mhlo.topk(%arg0, k = 2) : tensor<?x5x?xi1> -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>)
  %values, %indices = chlo.top_k(%arg0, k = 2) : tensor<?x5x?xi1> -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>)
  return %values, %indices : tensor<?x5x2xi1>, tensor<?x5x2xi32>
}

// -----

func.func @unranked_top_k(%arg : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi32>) {
  // expected-error@+1 {{failed to legalize operation 'chlo.top_k' that was explicitly marked illegal}}
  %1:2 = chlo.top_k(%arg, k=8) : tensor<*xf32> -> (tensor<*xf32>, tensor<*xi32>)
  func.return %1#0, %1#1 : tensor<*xf32>, tensor<*xi32>
}

// -----

// Verify bessel_i1e operator for f16, f32, f64 separately as they use
// different coefficients.

// CHECK-LABEL: @bessel_i1e_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16xf16>)
func.func @bessel_i1e_f16(%arg: tensor<16x16xf16>) -> tensor<16x16xf16> {
  // CHECK-NEXT:  %[[TMP_0:.*]] = mhlo.convert %[[ARG0]] : (tensor<16x16xf16>) -> tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_1:.*]] = mhlo.abs %[[TMP_0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_2:.*]] = mhlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_3:.*]] = mhlo.constant dense<2.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_4:.*]] = mhlo.constant dense<3.200000e+01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_5:.*]] = mhlo.constant dense<8.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_6:.*]] = mhlo.multiply %[[TMP_2]], %[[TMP_1]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_7:.*]] = mhlo.subtract %[[TMP_6]], %[[TMP_3]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_8:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_9:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_10:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_11:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_8]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_12:.*]] = mhlo.subtract %[[TMP_11]], %[[TMP_9]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_13:.*]] = mhlo.constant dense<9.38153732E-9> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_14:.*]] = mhlo.add %[[TMP_12]], %[[TMP_13]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_15:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_14]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_16:.*]] = mhlo.subtract %[[TMP_15]], %[[TMP_8]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_17:.*]] = mhlo.constant dense<-4.44505908E-8> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_18:.*]] = mhlo.add %[[TMP_16]], %[[TMP_17]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_19:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_18]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_20:.*]] = mhlo.subtract %[[TMP_19]], %[[TMP_14]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_21:.*]] = mhlo.constant dense<2.00329481E-7> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_22:.*]] = mhlo.add %[[TMP_20]], %[[TMP_21]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_23:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_22]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_24:.*]] = mhlo.subtract %[[TMP_23]], %[[TMP_18]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_25:.*]] = mhlo.constant dense<-8.568720e-07> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_26:.*]] = mhlo.add %[[TMP_24]], %[[TMP_25]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_27:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_26]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_28:.*]] = mhlo.subtract %[[TMP_27]], %[[TMP_22]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_29:.*]] = mhlo.constant dense<3.47025139E-6> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_30:.*]] = mhlo.add %[[TMP_28]], %[[TMP_29]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_31:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_30]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_32:.*]] = mhlo.subtract %[[TMP_31]], %[[TMP_26]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_33:.*]] = mhlo.constant dense<-1.32731639E-5> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_34:.*]] = mhlo.add %[[TMP_32]], %[[TMP_33]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_35:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_34]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_36:.*]] = mhlo.subtract %[[TMP_35]], %[[TMP_30]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_37:.*]] = mhlo.constant dense<4.78156508E-5> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_38:.*]] = mhlo.add %[[TMP_36]], %[[TMP_37]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_39:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_38]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_40:.*]] = mhlo.subtract %[[TMP_39]], %[[TMP_34]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_41:.*]] = mhlo.constant dense<-1.61760821E-4> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_42:.*]] = mhlo.add %[[TMP_40]], %[[TMP_41]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_43:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_42]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_44:.*]] = mhlo.subtract %[[TMP_43]], %[[TMP_38]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_45:.*]] = mhlo.constant dense<5.122860e-04> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_46:.*]] = mhlo.add %[[TMP_44]], %[[TMP_45]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_47:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_46]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_48:.*]] = mhlo.subtract %[[TMP_47]], %[[TMP_42]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_49:.*]] = mhlo.constant dense<-0.00151357241> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_50:.*]] = mhlo.add %[[TMP_48]], %[[TMP_49]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_51:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_50]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_52:.*]] = mhlo.subtract %[[TMP_51]], %[[TMP_46]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_53:.*]] = mhlo.constant dense<0.0041564228> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_54:.*]] = mhlo.add %[[TMP_52]], %[[TMP_53]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_55:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_54]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_56:.*]] = mhlo.subtract %[[TMP_55]], %[[TMP_50]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_57:.*]] = mhlo.constant dense<-0.0105640851> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_58:.*]] = mhlo.add %[[TMP_56]], %[[TMP_57]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_59:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_58]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_60:.*]] = mhlo.subtract %[[TMP_59]], %[[TMP_54]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_61:.*]] = mhlo.constant dense<0.0247264486> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_62:.*]] = mhlo.add %[[TMP_60]], %[[TMP_61]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_63:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_62]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_64:.*]] = mhlo.subtract %[[TMP_63]], %[[TMP_58]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_65:.*]] = mhlo.constant dense<-0.0529459827> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_66:.*]] = mhlo.add %[[TMP_64]], %[[TMP_65]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_67:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_66]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_68:.*]] = mhlo.subtract %[[TMP_67]], %[[TMP_62]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_69:.*]] = mhlo.constant dense<0.102643661> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_70:.*]] = mhlo.add %[[TMP_68]], %[[TMP_69]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_71:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_70]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_72:.*]] = mhlo.subtract %[[TMP_71]], %[[TMP_66]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_73:.*]] = mhlo.constant dense<-0.176416516> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_74:.*]] = mhlo.add %[[TMP_72]], %[[TMP_73]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_75:.*]] = mhlo.multiply %[[TMP_7]], %[[TMP_74]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_76:.*]] = mhlo.subtract %[[TMP_75]], %[[TMP_70]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_77:.*]] = mhlo.constant dense<0.252587199> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_78:.*]] = mhlo.add %[[TMP_76]], %[[TMP_77]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_79:.*]] = mhlo.subtract %[[TMP_78]], %[[TMP_70]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_80:.*]] = mhlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_81:.*]] = mhlo.multiply %[[TMP_79]], %[[TMP_80]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_82:.*]] = mhlo.multiply %[[TMP_1]], %[[TMP_81]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_83:.*]] = mhlo.divide %[[TMP_4]], %[[TMP_1]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_84:.*]] = mhlo.subtract %[[TMP_83]], %[[TMP_3]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_85:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_86:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_87:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_88:.*]] = mhlo.multiply %[[TMP_84]], %[[TMP_85]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_89:.*]] = mhlo.subtract %[[TMP_88]], %[[TMP_86]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_90:.*]] = mhlo.constant dense<-3.83538046E-9> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_91:.*]] = mhlo.add %[[TMP_89]], %[[TMP_90]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_92:.*]] = mhlo.multiply %[[TMP_84]], %[[TMP_91]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_93:.*]] = mhlo.subtract %[[TMP_92]], %[[TMP_85]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_94:.*]] = mhlo.constant dense<-2.63146891E-8> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_95:.*]] = mhlo.add %[[TMP_93]], %[[TMP_94]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_96:.*]] = mhlo.multiply %[[TMP_84]], %[[TMP_95]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_97:.*]] = mhlo.subtract %[[TMP_96]], %[[TMP_91]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_98:.*]] = mhlo.constant dense<-2.51223611E-7> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_99:.*]] = mhlo.add %[[TMP_97]], %[[TMP_98]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_100:.*]] = mhlo.multiply %[[TMP_84]], %[[TMP_99]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_101:.*]] = mhlo.subtract %[[TMP_100]], %[[TMP_95]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_102:.*]] = mhlo.constant dense<-3.88256467E-6> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_103:.*]] = mhlo.add %[[TMP_101]], %[[TMP_102]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_104:.*]] = mhlo.multiply %[[TMP_84]], %[[TMP_103]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_105:.*]] = mhlo.subtract %[[TMP_104]], %[[TMP_99]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_106:.*]] = mhlo.constant dense<-1.10588939E-4> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_107:.*]] = mhlo.add %[[TMP_105]], %[[TMP_106]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_108:.*]] = mhlo.multiply %[[TMP_84]], %[[TMP_107]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_109:.*]] = mhlo.subtract %[[TMP_108]], %[[TMP_103]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_110:.*]] = mhlo.constant dense<-0.00976109784> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_111:.*]] = mhlo.add %[[TMP_109]], %[[TMP_110]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_112:.*]] = mhlo.multiply %[[TMP_84]], %[[TMP_111]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_113:.*]] = mhlo.subtract %[[TMP_112]], %[[TMP_107]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_114:.*]] = mhlo.constant dense<0.778576254> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_115:.*]] = mhlo.add %[[TMP_113]], %[[TMP_114]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_116:.*]] = mhlo.subtract %[[TMP_115]], %[[TMP_107]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_117:.*]] = mhlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_118:.*]] = mhlo.multiply %[[TMP_116]], %[[TMP_117]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_119:.*]] = mhlo.sqrt %[[TMP_1]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_120:.*]] = mhlo.divide %[[TMP_118]], %[[TMP_119]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_121:.*]] = mhlo.compare LE, %[[TMP_1]], %[[TMP_5]] : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
  // CHECK-NEXT:  %[[TMP_122:.*]] = mhlo.select %[[TMP_121]], %[[TMP_82]], %[[TMP_120]] : tensor<16x16xi1>, tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_123:.*]] = mhlo.sign %[[TMP_0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_124:.*]] = mhlo.multiply %[[TMP_123]], %[[TMP_122]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_125:.*]] = mhlo.convert %[[TMP_124]] : (tensor<16x16xf32>) -> tensor<16x16xf16>
  // CHECK-NEXT:  return %[[TMP_125]] : tensor<16x16xf16>
  %0 = chlo.bessel_i1e %arg : tensor<16x16xf16> -> tensor<16x16xf16>
  func.return %0 : tensor<16x16xf16>
}

// -----

// CHECK-LABEL: @bessel_i1e_f32
// CHECK-SAME:   (%[[ARG0:.*]]: tensor<16x16xf32>)
func.func @bessel_i1e_f32(%arg : tensor<16x16xf32>) -> tensor<16x16xf32> {
  // CHECK-NEXT:  %[[TMP_0:.*]] = mhlo.abs %[[ARG0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_1:.*]] = mhlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_2:.*]] = mhlo.constant dense<2.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_3:.*]] = mhlo.constant dense<3.200000e+01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_4:.*]] = mhlo.constant dense<8.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_5:.*]] = mhlo.multiply %[[TMP_1]], %[[TMP_0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_6:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_2]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_7:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_8:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_9:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_10:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_7]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_11:.*]] = mhlo.subtract %[[TMP_10]], %[[TMP_8]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_12:.*]] = mhlo.constant dense<9.38153732E-9> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_13:.*]] = mhlo.add %[[TMP_11]], %[[TMP_12]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_14:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_13]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_15:.*]] = mhlo.subtract %[[TMP_14]], %[[TMP_7]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_16:.*]] = mhlo.constant dense<-4.44505908E-8> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_17:.*]] = mhlo.add %[[TMP_15]], %[[TMP_16]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_18:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_17]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_19:.*]] = mhlo.subtract %[[TMP_18]], %[[TMP_13]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_20:.*]] = mhlo.constant dense<2.00329481E-7> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_21:.*]] = mhlo.add %[[TMP_19]], %[[TMP_20]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_22:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_21]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_23:.*]] = mhlo.subtract %[[TMP_22]], %[[TMP_17]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_24:.*]] = mhlo.constant dense<-8.568720e-07> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_25:.*]] = mhlo.add %[[TMP_23]], %[[TMP_24]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_26:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_25]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_27:.*]] = mhlo.subtract %[[TMP_26]], %[[TMP_21]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_28:.*]] = mhlo.constant dense<3.47025139E-6> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_29:.*]] = mhlo.add %[[TMP_27]], %[[TMP_28]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_30:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_29]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_31:.*]] = mhlo.subtract %[[TMP_30]], %[[TMP_25]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_32:.*]] = mhlo.constant dense<-1.32731639E-5> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_33:.*]] = mhlo.add %[[TMP_31]], %[[TMP_32]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_34:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_33]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_35:.*]] = mhlo.subtract %[[TMP_34]], %[[TMP_29]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_36:.*]] = mhlo.constant dense<4.78156508E-5> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_37:.*]] = mhlo.add %[[TMP_35]], %[[TMP_36]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_38:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_37]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_39:.*]] = mhlo.subtract %[[TMP_38]], %[[TMP_33]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_40:.*]] = mhlo.constant dense<-1.61760821E-4> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_41:.*]] = mhlo.add %[[TMP_39]], %[[TMP_40]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_42:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_41]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_43:.*]] = mhlo.subtract %[[TMP_42]], %[[TMP_37]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_44:.*]] = mhlo.constant dense<5.122860e-04> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_45:.*]] = mhlo.add %[[TMP_43]], %[[TMP_44]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_46:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_45]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_47:.*]] = mhlo.subtract %[[TMP_46]], %[[TMP_41]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_48:.*]] = mhlo.constant dense<-0.00151357241> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_49:.*]] = mhlo.add %[[TMP_47]], %[[TMP_48]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_50:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_49]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_51:.*]] = mhlo.subtract %[[TMP_50]], %[[TMP_45]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_52:.*]] = mhlo.constant dense<0.0041564228> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_53:.*]] = mhlo.add %[[TMP_51]], %[[TMP_52]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_54:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_53]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_55:.*]] = mhlo.subtract %[[TMP_54]], %[[TMP_49]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_56:.*]] = mhlo.constant dense<-0.0105640851> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_57:.*]] = mhlo.add %[[TMP_55]], %[[TMP_56]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_58:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_57]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_59:.*]] = mhlo.subtract %[[TMP_58]], %[[TMP_53]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_60:.*]] = mhlo.constant dense<0.0247264486> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_61:.*]] = mhlo.add %[[TMP_59]], %[[TMP_60]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_62:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_61]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_63:.*]] = mhlo.subtract %[[TMP_62]], %[[TMP_57]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_64:.*]] = mhlo.constant dense<-0.0529459827> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_65:.*]] = mhlo.add %[[TMP_63]], %[[TMP_64]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_66:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_65]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_67:.*]] = mhlo.subtract %[[TMP_66]], %[[TMP_61]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_68:.*]] = mhlo.constant dense<0.102643661> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_69:.*]] = mhlo.add %[[TMP_67]], %[[TMP_68]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_70:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_69]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_71:.*]] = mhlo.subtract %[[TMP_70]], %[[TMP_65]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_72:.*]] = mhlo.constant dense<-0.176416516> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_73:.*]] = mhlo.add %[[TMP_71]], %[[TMP_72]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_74:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_73]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_75:.*]] = mhlo.subtract %[[TMP_74]], %[[TMP_69]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_76:.*]] = mhlo.constant dense<0.252587199> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_77:.*]] = mhlo.add %[[TMP_75]], %[[TMP_76]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_78:.*]] = mhlo.subtract %[[TMP_77]], %[[TMP_69]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_79:.*]] = mhlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_80:.*]] = mhlo.multiply %[[TMP_78]], %[[TMP_79]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_81:.*]] = mhlo.multiply %[[TMP_0]], %[[TMP_80]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_82:.*]] = mhlo.divide %[[TMP_3]], %[[TMP_0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_83:.*]] = mhlo.subtract %[[TMP_82]], %[[TMP_2]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_84:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_85:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_86:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_87:.*]] = mhlo.multiply %[[TMP_83]], %[[TMP_84]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_88:.*]] = mhlo.subtract %[[TMP_87]], %[[TMP_85]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_89:.*]] = mhlo.constant dense<-3.83538046E-9> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_90:.*]] = mhlo.add %[[TMP_88]], %[[TMP_89]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_91:.*]] = mhlo.multiply %[[TMP_83]], %[[TMP_90]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_92:.*]] = mhlo.subtract %[[TMP_91]], %[[TMP_84]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_93:.*]] = mhlo.constant dense<-2.63146891E-8> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_94:.*]] = mhlo.add %[[TMP_92]], %[[TMP_93]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_95:.*]] = mhlo.multiply %[[TMP_83]], %[[TMP_94]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_96:.*]] = mhlo.subtract %[[TMP_95]], %[[TMP_90]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_97:.*]] = mhlo.constant dense<-2.51223611E-7> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_98:.*]] = mhlo.add %[[TMP_96]], %[[TMP_97]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_99:.*]] = mhlo.multiply %[[TMP_83]], %[[TMP_98]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_100:.*]] = mhlo.subtract %[[TMP_99]], %[[TMP_94]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_101:.*]] = mhlo.constant dense<-3.88256467E-6> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_102:.*]] = mhlo.add %[[TMP_100]], %[[TMP_101]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_103:.*]] = mhlo.multiply %[[TMP_83]], %[[TMP_102]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_104:.*]] = mhlo.subtract %[[TMP_103]], %[[TMP_98]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_105:.*]] = mhlo.constant dense<-1.10588939E-4> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_106:.*]] = mhlo.add %[[TMP_104]], %[[TMP_105]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_107:.*]] = mhlo.multiply %[[TMP_83]], %[[TMP_106]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_108:.*]] = mhlo.subtract %[[TMP_107]], %[[TMP_102]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_109:.*]] = mhlo.constant dense<-0.00976109784> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_110:.*]] = mhlo.add %[[TMP_108]], %[[TMP_109]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_111:.*]] = mhlo.multiply %[[TMP_83]], %[[TMP_110]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_112:.*]] = mhlo.subtract %[[TMP_111]], %[[TMP_106]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_113:.*]] = mhlo.constant dense<0.778576254> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_114:.*]] = mhlo.add %[[TMP_112]], %[[TMP_113]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_115:.*]] = mhlo.subtract %[[TMP_114]], %[[TMP_106]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_116:.*]] = mhlo.constant dense<5.000000e-01> : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_117:.*]] = mhlo.multiply %[[TMP_115]], %[[TMP_116]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_118:.*]] = mhlo.sqrt %[[TMP_0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_119:.*]] = mhlo.divide %[[TMP_117]], %[[TMP_118]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_120:.*]] = mhlo.compare LE, %[[TMP_0]], %[[TMP_4]] : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
  // CHECK-NEXT:  %[[TMP_121:.*]] = mhlo.select %[[TMP_120]], %[[TMP_81]], %[[TMP_119]] : tensor<16x16xi1>, tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_122:.*]] = mhlo.sign %[[ARG0]] : tensor<16x16xf32>
  // CHECK-NEXT:  %[[TMP_123:.*]] = mhlo.multiply %[[TMP_122]], %[[TMP_121]] : tensor<16x16xf32>
  // CHECK-NEXT:  return %[[TMP_123]] : tensor<16x16xf32>
  %0 = chlo.bessel_i1e %arg : tensor<16x16xf32> -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: @bessel_i1e_f64
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16xf64>)
func.func @bessel_i1e_f64(%arg : tensor<16x16xf64>) -> tensor<16x16xf64> {
  // CHECK-NEXT: %[[TMP_0:.*]] = mhlo.abs %[[ARG0]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_1:.*]] = mhlo.constant dense<5.000000e-01> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_2:.*]] = mhlo.constant dense<2.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_3:.*]] = mhlo.constant dense<3.200000e+01> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_4:.*]] = mhlo.constant dense<8.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_5:.*]] = mhlo.multiply %[[TMP_1]], %[[TMP_0]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_6:.*]] = mhlo.subtract %[[TMP_5]], %[[TMP_2]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_7:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_8:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_9:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_10:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_7]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_11:.*]] = mhlo.subtract %[[TMP_10]], %[[TMP_8]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_12:.*]] = mhlo.constant dense<2.7779141127610464E-18> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_13:.*]] = mhlo.add %[[TMP_11]], %[[TMP_12]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_14:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_13]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_15:.*]] = mhlo.subtract %[[TMP_14]], %[[TMP_7]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_16:.*]] = mhlo.constant dense<-2.111421214358166E-17> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_17:.*]] = mhlo.add %[[TMP_15]], %[[TMP_16]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_18:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_17]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_19:.*]] = mhlo.subtract %[[TMP_18]], %[[TMP_13]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_20:.*]] = mhlo.constant dense<1.5536319577362005E-16> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_21:.*]] = mhlo.add %[[TMP_19]], %[[TMP_20]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_22:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_21]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_23:.*]] = mhlo.subtract %[[TMP_22]], %[[TMP_17]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_24:.*]] = mhlo.constant dense<-1.1055969477353862E-15> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_25:.*]] = mhlo.add %[[TMP_23]], %[[TMP_24]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_26:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_25]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_27:.*]] = mhlo.subtract %[[TMP_26]], %[[TMP_21]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_28:.*]] = mhlo.constant dense<7.6006842947354077E-15> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_29:.*]] = mhlo.add %[[TMP_27]], %[[TMP_28]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_30:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_29]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_31:.*]] = mhlo.subtract %[[TMP_30]], %[[TMP_25]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_32:.*]] = mhlo.constant dense<-5.0421855047279118E-14> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_33:.*]] = mhlo.add %[[TMP_31]], %[[TMP_32]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_34:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_33]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_35:.*]] = mhlo.subtract %[[TMP_34]], %[[TMP_29]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_36:.*]] = mhlo.constant dense<3.2237933659455748E-13> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_37:.*]] = mhlo.add %[[TMP_35]], %[[TMP_36]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_38:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_37]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_39:.*]] = mhlo.subtract %[[TMP_38]], %[[TMP_33]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_40:.*]] = mhlo.constant dense<-1.9839743977649436E-12> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_41:.*]] = mhlo.add %[[TMP_39]], %[[TMP_40]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_42:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_41]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_43:.*]] = mhlo.subtract %[[TMP_42]], %[[TMP_37]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_44:.*]] = mhlo.constant dense<1.1736186298890901E-11> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_45:.*]] = mhlo.add %[[TMP_43]], %[[TMP_44]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_46:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_45]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_47:.*]] = mhlo.subtract %[[TMP_46]], %[[TMP_41]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_48:.*]] = mhlo.constant dense<-6.6634897235020271E-11> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_49:.*]] = mhlo.add %[[TMP_47]], %[[TMP_48]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_50:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_49]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_51:.*]] = mhlo.subtract %[[TMP_50]], %[[TMP_45]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_52:.*]] = mhlo.constant dense<3.6255902815521172E-10> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_53:.*]] = mhlo.add %[[TMP_51]], %[[TMP_52]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_54:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_53]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_55:.*]] = mhlo.subtract %[[TMP_54]], %[[TMP_49]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_56:.*]] = mhlo.constant dense<-1.8872497517228294E-9> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_57:.*]] = mhlo.add %[[TMP_55]], %[[TMP_56]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_58:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_57]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_59:.*]] = mhlo.subtract %[[TMP_58]], %[[TMP_53]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_60:.*]] = mhlo.constant dense<9.3815373864957726E-9> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_61:.*]] = mhlo.add %[[TMP_59]], %[[TMP_60]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_62:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_61]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_63:.*]] = mhlo.subtract %[[TMP_62]], %[[TMP_57]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_64:.*]] = mhlo.constant dense<-4.4450591287963281E-8> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_65:.*]] = mhlo.add %[[TMP_63]], %[[TMP_64]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_66:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_65]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_67:.*]] = mhlo.subtract %[[TMP_66]], %[[TMP_61]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_68:.*]] = mhlo.constant dense<2.0032947535521353E-7> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_69:.*]] = mhlo.add %[[TMP_67]], %[[TMP_68]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_70:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_69]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_71:.*]] = mhlo.subtract %[[TMP_70]], %[[TMP_65]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_72:.*]] = mhlo.constant dense<-8.5687202646954547E-7> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_73:.*]] = mhlo.add %[[TMP_71]], %[[TMP_72]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_74:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_73]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_75:.*]] = mhlo.subtract %[[TMP_74]], %[[TMP_69]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_76:.*]] = mhlo.constant dense<3.4702513081376785E-6> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_77:.*]] = mhlo.add %[[TMP_75]], %[[TMP_76]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_78:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_77]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_79:.*]] = mhlo.subtract %[[TMP_78]], %[[TMP_73]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_80:.*]] = mhlo.constant dense<-1.3273163656039436E-5> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_81:.*]] = mhlo.add %[[TMP_79]], %[[TMP_80]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_82:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_81]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_83:.*]] = mhlo.subtract %[[TMP_82]], %[[TMP_77]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_84:.*]] = mhlo.constant dense<4.7815651075500542E-5> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_85:.*]] = mhlo.add %[[TMP_83]], %[[TMP_84]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_86:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_85]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_87:.*]] = mhlo.subtract %[[TMP_86]], %[[TMP_81]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_88:.*]] = mhlo.constant dense<-1.6176081582589674E-4> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_89:.*]] = mhlo.add %[[TMP_87]], %[[TMP_88]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_90:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_89]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_91:.*]] = mhlo.subtract %[[TMP_90]], %[[TMP_85]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_92:.*]] = mhlo.constant dense<5.1228595616857576E-4> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_93:.*]] = mhlo.add %[[TMP_91]], %[[TMP_92]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_94:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_93]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_95:.*]] = mhlo.subtract %[[TMP_94]], %[[TMP_89]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_96:.*]] = mhlo.constant dense<-0.0015135724506312532> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_97:.*]] = mhlo.add %[[TMP_95]], %[[TMP_96]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_98:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_97]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_99:.*]] = mhlo.subtract %[[TMP_98]], %[[TMP_93]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_100:.*]] = mhlo.constant dense<0.0041564229443128882> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_101:.*]] = mhlo.add %[[TMP_99]], %[[TMP_100]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_102:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_101]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_103:.*]] = mhlo.subtract %[[TMP_102]], %[[TMP_97]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_104:.*]] = mhlo.constant dense<-0.010564084894626197> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_105:.*]] = mhlo.add %[[TMP_103]], %[[TMP_104]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_106:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_105]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_107:.*]] = mhlo.subtract %[[TMP_106]], %[[TMP_101]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_108:.*]] = mhlo.constant dense<0.024726449030626516> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_109:.*]] = mhlo.add %[[TMP_107]], %[[TMP_108]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_110:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_109]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_111:.*]] = mhlo.subtract %[[TMP_110]], %[[TMP_105]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_112:.*]] = mhlo.constant dense<-0.052945981208094989> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_113:.*]] = mhlo.add %[[TMP_111]], %[[TMP_112]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_114:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_113]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_115:.*]] = mhlo.subtract %[[TMP_114]], %[[TMP_109]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_116:.*]] = mhlo.constant dense<0.10264365868984709> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_117:.*]] = mhlo.add %[[TMP_115]], %[[TMP_116]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_118:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_117]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_119:.*]] = mhlo.subtract %[[TMP_118]], %[[TMP_113]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_120:.*]] = mhlo.constant dense<-0.17641651835783406> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_121:.*]] = mhlo.add %[[TMP_119]], %[[TMP_120]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_122:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_121]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_123:.*]] = mhlo.subtract %[[TMP_122]], %[[TMP_117]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_124:.*]] = mhlo.constant dense<0.25258718644363365> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_125:.*]] = mhlo.add %[[TMP_123]], %[[TMP_124]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_126:.*]] = mhlo.subtract %[[TMP_125]], %[[TMP_117]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_127:.*]] = mhlo.constant dense<5.000000e-01> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_128:.*]] = mhlo.multiply %[[TMP_126]], %[[TMP_127]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_129:.*]] = mhlo.multiply %[[TMP_0]], %[[TMP_128]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_130:.*]] = mhlo.divide %[[TMP_3]], %[[TMP_0]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_131:.*]] = mhlo.subtract %[[TMP_130]], %[[TMP_2]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_132:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_133:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_134:.*]] = mhlo.constant dense<0.000000e+00> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_135:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_132]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_136:.*]] = mhlo.subtract %[[TMP_135]], %[[TMP_133]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_137:.*]] = mhlo.constant dense<7.5172963108421052E-18> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_138:.*]] = mhlo.add %[[TMP_136]], %[[TMP_137]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_139:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_138]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_140:.*]] = mhlo.subtract %[[TMP_139]], %[[TMP_132]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_141:.*]] = mhlo.constant dense<4.4143483230717077E-18> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_142:.*]] = mhlo.add %[[TMP_140]], %[[TMP_141]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_143:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_142]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_144:.*]] = mhlo.subtract %[[TMP_143]], %[[TMP_138]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_145:.*]] = mhlo.constant dense<-4.6503053684893586E-17> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_146:.*]] = mhlo.add %[[TMP_144]], %[[TMP_145]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_147:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_146]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_148:.*]] = mhlo.subtract %[[TMP_147]], %[[TMP_142]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_149:.*]] = mhlo.constant dense<-3.2095259219934238E-17> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_150:.*]] = mhlo.add %[[TMP_148]], %[[TMP_149]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_151:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_150]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_152:.*]] = mhlo.subtract %[[TMP_151]], %[[TMP_146]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_153:.*]] = mhlo.constant dense<2.9626289976459501E-16> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_154:.*]] = mhlo.add %[[TMP_152]], %[[TMP_153]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_155:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_154]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_156:.*]] = mhlo.subtract %[[TMP_155]], %[[TMP_150]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_157:.*]] = mhlo.constant dense<3.3082023109209285E-16> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_158:.*]] = mhlo.add %[[TMP_156]], %[[TMP_157]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_159:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_158]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_160:.*]] = mhlo.subtract %[[TMP_159]], %[[TMP_154]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_161:.*]] = mhlo.constant dense<-1.8803547755107825E-15> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_162:.*]] = mhlo.add %[[TMP_160]], %[[TMP_161]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_163:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_162]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_164:.*]] = mhlo.subtract %[[TMP_163]], %[[TMP_158]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_165:.*]] = mhlo.constant dense<-3.8144030724370075E-15> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_166:.*]] = mhlo.add %[[TMP_164]], %[[TMP_165]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_167:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_166]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_168:.*]] = mhlo.subtract %[[TMP_167]], %[[TMP_162]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_169:.*]] = mhlo.constant dense<1.0420276984128802E-14> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_170:.*]] = mhlo.add %[[TMP_168]], %[[TMP_169]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_171:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_170]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_172:.*]] = mhlo.subtract %[[TMP_171]], %[[TMP_166]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_173:.*]] = mhlo.constant dense<4.272440016711951E-14> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_174:.*]] = mhlo.add %[[TMP_172]], %[[TMP_173]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_175:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_174]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_176:.*]] = mhlo.subtract %[[TMP_175]], %[[TMP_170]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_177:.*]] = mhlo.constant dense<-2.1015418427726643E-14> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_178:.*]] = mhlo.add %[[TMP_176]], %[[TMP_177]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_179:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_178]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_180:.*]] = mhlo.subtract %[[TMP_179]], %[[TMP_174]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_181:.*]] = mhlo.constant dense<-4.0835511110921974E-13> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_182:.*]] = mhlo.add %[[TMP_180]], %[[TMP_181]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_183:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_182]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_184:.*]] = mhlo.subtract %[[TMP_183]], %[[TMP_178]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_185:.*]] = mhlo.constant dense<-7.1985517762459084E-13> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_186:.*]] = mhlo.add %[[TMP_184]], %[[TMP_185]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_187:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_186]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_188:.*]] = mhlo.subtract %[[TMP_187]], %[[TMP_182]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_189:.*]] = mhlo.constant dense<2.0356285441470896E-12> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_190:.*]] = mhlo.add %[[TMP_188]], %[[TMP_189]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_191:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_190]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_192:.*]] = mhlo.subtract %[[TMP_191]], %[[TMP_186]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_193:.*]] = mhlo.constant dense<1.4125807436613782E-11> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_194:.*]] = mhlo.add %[[TMP_192]], %[[TMP_193]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_195:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_194]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_196:.*]] = mhlo.subtract %[[TMP_195]], %[[TMP_190]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_197:.*]] = mhlo.constant dense<3.2526035830154884E-11> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_198:.*]] = mhlo.add %[[TMP_196]], %[[TMP_197]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_199:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_198]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_200:.*]] = mhlo.subtract %[[TMP_199]], %[[TMP_194]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_201:.*]] = mhlo.constant dense<-1.8974958123505413E-11> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_202:.*]] = mhlo.add %[[TMP_200]], %[[TMP_201]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_203:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_202]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_204:.*]] = mhlo.subtract %[[TMP_203]], %[[TMP_198]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_205:.*]] = mhlo.constant dense<-5.5897434621965838E-10> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_206:.*]] = mhlo.add %[[TMP_204]], %[[TMP_205]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_207:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_206]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_208:.*]] = mhlo.subtract %[[TMP_207]], %[[TMP_202]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_209:.*]] = mhlo.constant dense<-3.835380385964237E-9> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_210:.*]] = mhlo.add %[[TMP_208]], %[[TMP_209]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_211:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_210]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_212:.*]] = mhlo.subtract %[[TMP_211]], %[[TMP_206]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_213:.*]] = mhlo.constant dense<-2.6314688468895196E-8> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_214:.*]] = mhlo.add %[[TMP_212]], %[[TMP_213]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_215:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_214]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_216:.*]] = mhlo.subtract %[[TMP_215]], %[[TMP_210]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_217:.*]] = mhlo.constant dense<-2.5122362378702088E-7> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_218:.*]] = mhlo.add %[[TMP_216]], %[[TMP_217]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_219:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_218]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_220:.*]] = mhlo.subtract %[[TMP_219]], %[[TMP_214]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_221:.*]] = mhlo.constant dense<-3.8825648088776906E-6> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_222:.*]] = mhlo.add %[[TMP_220]], %[[TMP_221]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_223:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_222]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_224:.*]] = mhlo.subtract %[[TMP_223]], %[[TMP_218]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_225:.*]] = mhlo.constant dense<-1.1058893876262371E-4> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_226:.*]] = mhlo.add %[[TMP_224]], %[[TMP_225]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_227:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_226]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_228:.*]] = mhlo.subtract %[[TMP_227]], %[[TMP_222]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_229:.*]] = mhlo.constant dense<-0.0097610974913614687> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_230:.*]] = mhlo.add %[[TMP_228]], %[[TMP_229]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_231:.*]] = mhlo.multiply %[[TMP_131]], %[[TMP_230]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_232:.*]] = mhlo.subtract %[[TMP_231]], %[[TMP_226]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_233:.*]] = mhlo.constant dense<0.7785762350182801> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_234:.*]] = mhlo.add %[[TMP_232]], %[[TMP_233]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_235:.*]] = mhlo.subtract %[[TMP_234]], %[[TMP_226]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_236:.*]] = mhlo.constant dense<5.000000e-01> : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_237:.*]] = mhlo.multiply %[[TMP_235]], %[[TMP_236]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_238:.*]] = mhlo.sqrt %[[TMP_0]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_239:.*]] = mhlo.divide %[[TMP_237]], %[[TMP_238]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_240:.*]] = mhlo.compare LE, %[[TMP_0]], %[[TMP_4]] : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xi1>
  // CHECK-NEXT: %[[TMP_241:.*]] = mhlo.select %[[TMP_240]], %[[TMP_129]], %[[TMP_239]] : tensor<16x16xi1>, tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_242:.*]] = mhlo.sign %[[ARG0]] : tensor<16x16xf64>
  // CHECK-NEXT: %[[TMP_243:.*]] = mhlo.multiply %[[TMP_242]], %[[TMP_241]] : tensor<16x16xf64>
  // CHECK-NEXT: return %[[TMP_243]] : tensor<16x16xf64>
  %0 = chlo.bessel_i1e %arg : tensor<16x16xf64> -> tensor<16x16xf64>
  func.return %0 : tensor<16x16xf64>
}

// CHECK:  func.func @erf_inv([[ARG_0:%.*]]: tensor<16x16xf32>) {
// CHECK-DAG:     [[VAL_0:%.*]] = mhlo.negate [[ARG_0]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_1:%.*]] = mhlo.multiply [[ARG_0]], [[VAL_0]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_2:%.*]] = mhlo.log_plus_one [[VAL_1]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_3:%.*]] = mhlo.negate [[VAL_2]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_4:%.*]] = mhlo.constant dense<5.000000e+00> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_5:%.*]] = mhlo.compare  LT, [[VAL_3]], [[VAL_4]] : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
// CHECK-DAG:     [[VAL_6:%.*]] = mhlo.constant dense<2.500000e+00> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_7:%.*]] = mhlo.subtract [[VAL_3]], [[VAL_6]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_8:%.*]] = mhlo.sqrt [[VAL_3]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_9:%.*]] = mhlo.constant dense<3.000000e+00> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_10:%.*]] = mhlo.subtract [[VAL_8]], [[VAL_9]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_11:%.*]] = mhlo.select [[VAL_5]], [[VAL_7]], [[VAL_10]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_12:%.*]] = mhlo.constant dense<2.81022636E-8> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_13:%.*]] = mhlo.constant dense<-2.00214257E-4> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_14:%.*]] = mhlo.select [[VAL_5]], [[VAL_12]], [[VAL_13]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_15:%.*]] = mhlo.constant dense<3.43273939E-7> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_16:%.*]] = mhlo.constant dense<1.00950558E-4> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_17:%.*]] = mhlo.select [[VAL_5]], [[VAL_15]], [[VAL_16]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_18:%.*]] = mhlo.multiply [[VAL_14]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_19:%.*]] = mhlo.add [[VAL_17]], [[VAL_18]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_20:%.*]] = mhlo.constant dense<-3.5233877E-6> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_21:%.*]] = mhlo.constant dense<0.00134934322> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_22:%.*]] = mhlo.select [[VAL_5]], [[VAL_20]], [[VAL_21]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_23:%.*]] = mhlo.multiply [[VAL_19]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_24:%.*]] = mhlo.add [[VAL_22]], [[VAL_23]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_25:%.*]] = mhlo.constant dense<-4.39150654E-6> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_26:%.*]] = mhlo.constant dense<-0.00367342844> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_27:%.*]] = mhlo.select [[VAL_5]], [[VAL_25]], [[VAL_26]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_28:%.*]] = mhlo.multiply [[VAL_24]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_29:%.*]] = mhlo.add [[VAL_27]], [[VAL_28]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_30:%.*]] = mhlo.constant dense<2.1858087E-4> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_31:%.*]] = mhlo.constant dense<0.00573950773> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_32:%.*]] = mhlo.select [[VAL_5]], [[VAL_30]], [[VAL_31]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_33:%.*]] = mhlo.multiply [[VAL_29]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_34:%.*]] = mhlo.add [[VAL_32]], [[VAL_33]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_35:%.*]] = mhlo.constant dense<-0.00125372503> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_36:%.*]] = mhlo.constant dense<-0.0076224613> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_37:%.*]] = mhlo.select [[VAL_5]], [[VAL_35]], [[VAL_36]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_38:%.*]] = mhlo.multiply [[VAL_34]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_39:%.*]] = mhlo.add [[VAL_37]], [[VAL_38]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_40:%.*]] = mhlo.constant dense<-0.00417768164> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_41:%.*]] = mhlo.constant dense<0.00943887047> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_42:%.*]] = mhlo.select [[VAL_5]], [[VAL_40]], [[VAL_41]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_43:%.*]] = mhlo.multiply [[VAL_39]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_44:%.*]] = mhlo.add [[VAL_42]], [[VAL_43]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_45:%.*]] = mhlo.constant dense<0.246640727> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_46:%.*]] = mhlo.constant dense<1.00167406> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_47:%.*]] = mhlo.select [[VAL_5]], [[VAL_45]], [[VAL_46]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_48:%.*]] = mhlo.multiply [[VAL_44]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_49:%.*]] = mhlo.add [[VAL_47]], [[VAL_48]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_50:%.*]] = mhlo.constant dense<1.50140941> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_51:%.*]] = mhlo.constant dense<2.83297682> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_52:%.*]] = mhlo.select [[VAL_5]], [[VAL_50]], [[VAL_51]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-DAG:     [[VAL_53:%.*]] = mhlo.multiply [[VAL_49]], [[VAL_11]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_54:%.*]] = mhlo.add [[VAL_52]], [[VAL_53]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_55:%.*]] = mhlo.multiply [[VAL_54]], [[ARG_0]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_56:%.*]] = mhlo.abs [[ARG_0]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_57:%.*]] = mhlo.constant dense<1.000000e+00> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_58:%.*]] = mhlo.compare  EQ, [[VAL_56]], [[VAL_57]] : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
// CHECK-DAG:     [[VAL_59:%.*]] = mhlo.constant dense<0x7F800000> : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_60:%.*]] = mhlo.multiply [[ARG_0]], [[VAL_59]] : tensor<16x16xf32>
// CHECK-DAG:     [[VAL_61:%.*]] = mhlo.select [[VAL_58]], [[VAL_60]], [[VAL_55]] : tensor<16x16xi1>, tensor<16x16xf32>
func.func @erf_inv(%arg0 : tensor<16x16xf32>) {
  %0 = chlo.erf_inv %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
  return
}

// CHECK:  func.func @erf_inv_wide([[ARG_0:%.*]]: tensor<16x16xf64>) {
// CHECK-DAG:     [[VAL_0:%.*]] = mhlo.negate [[ARG_0]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_1:%.*]] = mhlo.multiply [[ARG_0]], [[VAL_0]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_2:%.*]] = mhlo.log_plus_one [[VAL_1]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_3:%.*]] = mhlo.negate [[VAL_2]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_4:%.*]] = mhlo.constant dense<6.250000e+00> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_5:%.*]] = mhlo.compare  LT, [[VAL_3]], [[VAL_4]] : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xi1>
// CHECK-DAG:     [[VAL_6:%.*]] = mhlo.constant dense<1.600000e+01> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_7:%.*]] = mhlo.compare  LT, [[VAL_3]], [[VAL_6]] : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xi1>
// CHECK-DAG:     [[VAL_8:%.*]] = mhlo.sqrt [[VAL_3]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_9:%.*]] = mhlo.constant dense<3.125000e+00> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_10:%.*]] = mhlo.subtract [[VAL_3]], [[VAL_9]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_11:%.*]] = mhlo.constant dense<3.250000e+00> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_12:%.*]] = mhlo.constant dense<5.000000e+00> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_13:%.*]] = mhlo.select [[VAL_7]], [[VAL_11]], [[VAL_12]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_14:%.*]] = mhlo.subtract [[VAL_8]], [[VAL_13]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_15:%.*]] = mhlo.select [[VAL_5]], [[VAL_10]], [[VAL_14]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_16:%.*]] = mhlo.constant dense<-3.6444120640178197E-21> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_17:%.*]] = mhlo.constant dense<2.2137376921775787E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_18:%.*]] = mhlo.select [[VAL_5]], [[VAL_16]], [[VAL_17]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_19:%.*]] = mhlo.constant dense<-2.7109920616438573E-11> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_20:%.*]] = mhlo.select [[VAL_7]], [[VAL_18]], [[VAL_19]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_21:%.*]] = mhlo.constant dense<-1.6850591381820166E-19> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_22:%.*]] = mhlo.constant dense<9.075656193888539E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_23:%.*]] = mhlo.select [[VAL_5]], [[VAL_21]], [[VAL_22]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_24:%.*]] = mhlo.constant dense<-2.5556418169965252E-10> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_25:%.*]] = mhlo.select [[VAL_7]], [[VAL_23]], [[VAL_24]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_26:%.*]] = mhlo.multiply [[VAL_20]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_27:%.*]] = mhlo.add [[VAL_25]], [[VAL_26]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_28:%.*]] = mhlo.constant dense<1.28584807152564E-18> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_29:%.*]] = mhlo.constant dense<-2.7517406297064545E-7> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_30:%.*]] = mhlo.select [[VAL_5]], [[VAL_28]], [[VAL_29]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_31:%.*]] = mhlo.constant dense<1.5076572693500548E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_32:%.*]] = mhlo.select [[VAL_7]], [[VAL_30]], [[VAL_31]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_33:%.*]] = mhlo.multiply [[VAL_27]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_34:%.*]] = mhlo.add [[VAL_32]], [[VAL_33]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_35:%.*]] = mhlo.constant dense<1.1157877678025181E-17> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_36:%.*]] = mhlo.constant dense<1.8239629214389228E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_37:%.*]] = mhlo.select [[VAL_5]], [[VAL_35]], [[VAL_36]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_38:%.*]] = mhlo.constant dense<-3.789465440126737E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_39:%.*]] = mhlo.select [[VAL_7]], [[VAL_37]], [[VAL_38]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_40:%.*]] = mhlo.multiply [[VAL_34]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_41:%.*]] = mhlo.add [[VAL_39]], [[VAL_40]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_42:%.*]] = mhlo.constant dense<-1.3331716628546209E-16> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_43:%.*]] = mhlo.constant dense<1.5027403968909828E-6> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_44:%.*]] = mhlo.select [[VAL_5]], [[VAL_42]], [[VAL_43]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_45:%.*]] = mhlo.constant dense<7.6157012080783394E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_46:%.*]] = mhlo.select [[VAL_7]], [[VAL_44]], [[VAL_45]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_47:%.*]] = mhlo.multiply [[VAL_41]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_48:%.*]] = mhlo.add [[VAL_46]], [[VAL_47]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_49:%.*]] = mhlo.constant dense<2.0972767875968562E-17> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_50:%.*]] = mhlo.constant dense<-4.013867526981546E-6> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_51:%.*]] = mhlo.select [[VAL_5]], [[VAL_49]], [[VAL_50]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_52:%.*]] = mhlo.constant dense<-1.496002662714924E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_53:%.*]] = mhlo.select [[VAL_7]], [[VAL_51]], [[VAL_52]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_54:%.*]] = mhlo.multiply [[VAL_48]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_55:%.*]] = mhlo.add [[VAL_53]], [[VAL_54]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_56:%.*]] = mhlo.constant dense<6.6376381343583238E-15> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_57:%.*]] = mhlo.constant dense<2.9234449089955446E-6> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_58:%.*]] = mhlo.select [[VAL_5]], [[VAL_56]], [[VAL_57]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_59:%.*]] = mhlo.constant dense<2.9147953450901081E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_60:%.*]] = mhlo.select [[VAL_7]], [[VAL_58]], [[VAL_59]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_61:%.*]] = mhlo.multiply [[VAL_55]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_62:%.*]] = mhlo.add [[VAL_60]], [[VAL_61]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_63:%.*]] = mhlo.constant dense<-4.0545662729752069E-14> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_64:%.*]] = mhlo.constant dense<1.2475304481671779E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_65:%.*]] = mhlo.select [[VAL_5]], [[VAL_63]], [[VAL_64]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_66:%.*]] = mhlo.constant dense<-6.7711997758452339E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_67:%.*]] = mhlo.select [[VAL_7]], [[VAL_65]], [[VAL_66]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_68:%.*]] = mhlo.multiply [[VAL_62]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_69:%.*]] = mhlo.add [[VAL_67]], [[VAL_68]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_70:%.*]] = mhlo.constant dense<-8.1519341976054721E-14> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_71:%.*]] = mhlo.constant dense<-4.7318229009055734E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_72:%.*]] = mhlo.select [[VAL_5]], [[VAL_70]], [[VAL_71]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_73:%.*]] = mhlo.constant dense<2.2900482228026655E-7> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_74:%.*]] = mhlo.select [[VAL_7]], [[VAL_72]], [[VAL_73]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_75:%.*]] = mhlo.multiply [[VAL_69]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_76:%.*]] = mhlo.add [[VAL_74]], [[VAL_75]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_77:%.*]] = mhlo.constant dense<2.6335093153082323E-12> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_78:%.*]] = mhlo.constant dense<6.8284851459573175E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_79:%.*]] = mhlo.select [[VAL_5]], [[VAL_77]], [[VAL_78]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_80:%.*]] = mhlo.constant dense<-9.9298272942317003E-7> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_81:%.*]] = mhlo.select [[VAL_7]], [[VAL_79]], [[VAL_80]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_82:%.*]] = mhlo.multiply [[VAL_76]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_83:%.*]] = mhlo.add [[VAL_81]], [[VAL_82]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_84:%.*]] = mhlo.constant dense<-1.2975133253453532E-11> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_85:%.*]] = mhlo.constant dense<2.4031110387097894E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_86:%.*]] = mhlo.select [[VAL_5]], [[VAL_84]], [[VAL_85]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_87:%.*]] = mhlo.constant dense<4.5260625972231537E-6> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_88:%.*]] = mhlo.select [[VAL_7]], [[VAL_86]], [[VAL_87]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_89:%.*]] = mhlo.multiply [[VAL_83]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_90:%.*]] = mhlo.add [[VAL_88]], [[VAL_89]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_91:%.*]] = mhlo.constant dense<-5.4154120542946279E-11> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_92:%.*]] = mhlo.constant dense<-3.5503752036284748E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_93:%.*]] = mhlo.select [[VAL_5]], [[VAL_91]], [[VAL_92]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_94:%.*]] = mhlo.constant dense<-1.9681778105531671E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_95:%.*]] = mhlo.select [[VAL_7]], [[VAL_93]], [[VAL_94]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_96:%.*]] = mhlo.multiply [[VAL_90]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_97:%.*]] = mhlo.add [[VAL_95]], [[VAL_96]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_98:%.*]] = mhlo.constant dense<1.0512122733215323E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_99:%.*]] = mhlo.constant dense<9.5328937973738049E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_100:%.*]] = mhlo.select [[VAL_5]], [[VAL_98]], [[VAL_99]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_101:%.*]] = mhlo.constant dense<7.5995277030017761E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_102:%.*]] = mhlo.select [[VAL_7]], [[VAL_100]], [[VAL_101]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_103:%.*]] = mhlo.multiply [[VAL_97]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_104:%.*]] = mhlo.add [[VAL_102]], [[VAL_103]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_105:%.*]] = mhlo.constant dense<-4.1126339803469837E-9> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_106:%.*]] = mhlo.constant dense<-0.0016882755560235047> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_107:%.*]] = mhlo.select [[VAL_5]], [[VAL_105]], [[VAL_106]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_108:%.*]] = mhlo.constant dense<-2.1503011930044477E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_109:%.*]] = mhlo.select [[VAL_7]], [[VAL_107]], [[VAL_108]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_110:%.*]] = mhlo.multiply [[VAL_104]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_111:%.*]] = mhlo.add [[VAL_109]], [[VAL_110]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_112:%.*]] = mhlo.constant dense<-2.9070369957882005E-8> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_113:%.*]] = mhlo.constant dense<0.0024914420961078508> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_114:%.*]] = mhlo.select [[VAL_5]], [[VAL_112]], [[VAL_113]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_115:%.*]] = mhlo.constant dense<-1.3871931833623122E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_116:%.*]] = mhlo.select [[VAL_7]], [[VAL_114]], [[VAL_115]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_117:%.*]] = mhlo.multiply [[VAL_111]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_118:%.*]] = mhlo.add [[VAL_116]], [[VAL_117]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_119:%.*]] = mhlo.constant dense<4.2347877827932404E-7> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_120:%.*]] = mhlo.constant dense<-0.0037512085075692412> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_121:%.*]] = mhlo.select [[VAL_5]], [[VAL_119]], [[VAL_120]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_122:%.*]] = mhlo.constant dense<1.0103004648645344> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_123:%.*]] = mhlo.select [[VAL_7]], [[VAL_121]], [[VAL_122]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_124:%.*]] = mhlo.multiply [[VAL_118]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_125:%.*]] = mhlo.add [[VAL_123]], [[VAL_124]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_126:%.*]] = mhlo.constant dense<-1.3654692000834679E-6> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_127:%.*]] = mhlo.constant dense<0.0053709145535900636> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_128:%.*]] = mhlo.select [[VAL_5]], [[VAL_126]], [[VAL_127]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_129:%.*]] = mhlo.constant dense<4.8499064014085844> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_130:%.*]] = mhlo.select [[VAL_7]], [[VAL_128]], [[VAL_129]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_131:%.*]] = mhlo.multiply [[VAL_125]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_132:%.*]] = mhlo.add [[VAL_130]], [[VAL_131]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_133:%.*]] = mhlo.constant dense<-1.3882523362786469E-5> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_134:%.*]] = mhlo.constant dense<1.0052589676941592> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_135:%.*]] = mhlo.select [[VAL_5]], [[VAL_133]], [[VAL_134]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_136:%.*]] = mhlo.multiply [[VAL_132]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_137:%.*]] = mhlo.add [[VAL_135]], [[VAL_136]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_138:%.*]] = mhlo.select [[VAL_7]], [[VAL_137]], [[VAL_132]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_139:%.*]] = mhlo.constant dense<1.8673420803405714E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_140:%.*]] = mhlo.constant dense<3.0838856104922208> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_141:%.*]] = mhlo.select [[VAL_5]], [[VAL_139]], [[VAL_140]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_142:%.*]] = mhlo.multiply [[VAL_138]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_143:%.*]] = mhlo.add [[VAL_141]], [[VAL_142]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_144:%.*]] = mhlo.select [[VAL_7]], [[VAL_143]], [[VAL_138]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_145:%.*]] = mhlo.constant dense<-7.4070253416626698E-4> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_146:%.*]] = mhlo.multiply [[VAL_144]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_147:%.*]] = mhlo.add [[VAL_145]], [[VAL_146]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_148:%.*]] = mhlo.select [[VAL_5]], [[VAL_147]], [[VAL_144]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_149:%.*]] = mhlo.constant dense<-0.0060336708714301491> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_150:%.*]] = mhlo.multiply [[VAL_148]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_151:%.*]] = mhlo.add [[VAL_149]], [[VAL_150]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_152:%.*]] = mhlo.select [[VAL_5]], [[VAL_151]], [[VAL_148]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_153:%.*]] = mhlo.constant dense<0.24015818242558962> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_154:%.*]] = mhlo.multiply [[VAL_152]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_155:%.*]] = mhlo.add [[VAL_153]], [[VAL_154]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_156:%.*]] = mhlo.select [[VAL_5]], [[VAL_155]], [[VAL_152]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_157:%.*]] = mhlo.constant dense<1.6536545626831027> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_158:%.*]] = mhlo.multiply [[VAL_156]], [[VAL_15]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_159:%.*]] = mhlo.add [[VAL_157]], [[VAL_158]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_160:%.*]] = mhlo.select [[VAL_5]], [[VAL_159]], [[VAL_156]] : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK-DAG:     [[VAL_161:%.*]] = mhlo.multiply [[VAL_160]], [[ARG_0]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_162:%.*]] = mhlo.abs [[ARG_0]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_163:%.*]] = mhlo.constant dense<1.000000e+00> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_164:%.*]] = mhlo.compare  EQ, [[VAL_162]], [[VAL_163]] : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xi1>
// CHECK-DAG:     [[VAL_165:%.*]] = mhlo.constant dense<0x7FF0000000000000> : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_166:%.*]] = mhlo.multiply [[ARG_0]], [[VAL_165]] : tensor<16x16xf64>
// CHECK-DAG:     [[VAL_167:%.*]] = mhlo.select [[VAL_164]], [[VAL_166]], [[VAL_161]] : tensor<16x16xi1>, tensor<16x16xf64>
func.func @erf_inv_wide(%arg0 : tensor<16x16xf64>) {
  %0 = chlo.erf_inv %arg0 : tensor<16x16xf64> -> tensor<16x16xf64>
  return
}

// -----

func.func @ragged_dot_non_contracting(%lhs : tensor<2x11x5xf32>, %rhs : tensor<3x2x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<2x11x7xf32> {
  // CHECK-HIGH-LEVEL: mhlo.ragged_dot
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2],
      lhs_ragged_dimensions = [1],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  func.return %0 : tensor<2x11x7xf32>
}

// -----

func.func @ragged_dot_contracting(%lhs : tensor<2x11x5xf32>, %rhs : tensor<2x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<3x2x11x7xf32> {
  // CHECK-HIGH-LEVEL: mhlo.ragged_dot
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [2],
      rhs_group_dimensions = []
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<2x11x5xf32>, tensor<2x5x7xf32>, tensor<3xi64>) -> tensor<3x2x11x7xf32>
  func.return %0 : tensor<3x2x11x7xf32>
}

// -----

func.func @ragged_dot_batch(%lhs : tensor<3x11x5xf32>, %rhs : tensor<3x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<3x11x7xf32> {
  // CHECK-HIGH-LEVEL: mhlo.ragged_dot
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [0],
      rhs_group_dimensions = []
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<3x11x5xf32>, tensor<3x5x7xf32>, tensor<3xi64>) -> tensor<3x11x7xf32>
  func.return %0 : tensor<3x11x7xf32>
}
