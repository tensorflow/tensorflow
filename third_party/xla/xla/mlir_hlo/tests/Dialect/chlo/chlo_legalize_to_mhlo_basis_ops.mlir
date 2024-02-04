// RUN: mlir-hlo-opt --chlo-legalize-to-hlo-basis-ops --chlo-legalize-to-hlo --split-input-file -verify-diagnostics %s | FileCheck %s 

// -----

// CHECK-LABEL: @erf_f64
// CHECK-SAME: %[[ARG:.*]]: tensor<f64>
func.func @erf_f64(%arg : tensor<f64>) -> tensor<f64> {
  // CHECK: %[[TMP_0:.*]] = mhlo.multiply %[[ARG]], %[[ARG]]
  // CHECK: %[[TMP_3:.*]] = mhlo.constant dense<9.6049737398705161>
  // CHECK: %[[TMP_5:.*]] = mhlo.multiply %[[TMP_3]], %[[TMP_0]]
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
  // CHECK: %[[TMP_20:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_22:.*]] = mhlo.multiply %[[TMP_20]], %[[TMP_0]]
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
  // CHECK: %[[TMP_40:.*]] = mhlo.negate %[[TMP_39]]
  // CHECK: %[[TMP_41:.*]] = mhlo.exponential %[[TMP_40]]
  // CHECK: %[[TMP_42:.*]] = mhlo.abs %[[ARG]]
  // CHECK: %[[TMP_45:.*]] = mhlo.constant dense<2.4619698147353052E-10>
  // CHECK: %[[TMP_47:.*]] = mhlo.multiply %[[TMP_45]], %[[TMP_42]]
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
  // CHECK: %[[TMP_74:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_76:.*]] = mhlo.multiply %[[TMP_74]], %[[TMP_42]]
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
  // CHECK: %[[TMP_103:.*]] = mhlo.constant dense<0.56418958354775506>
  // CHECK: %[[TMP_105:.*]] = mhlo.multiply %[[TMP_103]], %[[TMP_42]]
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
  // CHECK: %[[TMP_123:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[TMP_125:.*]] = mhlo.multiply %[[TMP_123]], %[[TMP_42]]
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
  // CHECK: %[[TMP_145:.*]] = mhlo.compare LT, %[[TMP_42]], %[[TMP_144]], NOTYPE
  // CHECK: %[[TMP_146:.*]] = mhlo.select %[[TMP_145]], %[[TMP_100]], %[[TMP_143]]
  // CHECK: %[[TMP_147:.*]] = mhlo.constant dense<-709.78271289338397>
  // CHECK: %[[TMP_148:.*]] = mhlo.compare LT, %[[TMP_40]], %[[TMP_147]], NOTYPE
  // CHECK: %[[TMP_149:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[TMP_150:.*]] = mhlo.select %[[TMP_148]], %[[TMP_149]], %[[TMP_146]]
  // CHECK: %[[TMP_152:.*]] = mhlo.compare LT, %[[ARG]], %[[TMP_149]], NOTYPE
  // CHECK: %[[TMP_153:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[TMP_154:.*]] = mhlo.subtract %[[TMP_153]], %[[TMP_150]]
  // CHECK: %[[TMP_155:.*]] = mhlo.select %[[TMP_152]], %[[TMP_154]], %[[TMP_150]]
  // CHECK: %[[TMP_156:.*]] = mhlo.subtract %[[TMP_38]], %[[TMP_155]]
  // CHECK: %[[TMP_157:.*]] = mhlo.abs %[[ARG]]
  // CHECK: %[[TMP_159:.*]] = mhlo.compare LT, %[[TMP_157]], %[[TMP_38]], NOTYPE
  // CHECK: %[[RESULT:.*]] = mhlo.select %[[TMP_159]], %[[TMP_37]], %[[TMP_156]]
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @erf_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @erf_f32(%arg : tensor<f32>) -> tensor<f32> {
  // CHECK-DAG: %[[TMP_0:.*]] = mhlo.constant dense<-4.000000e+00>
  // CHECK-DAG: %[[TMP_1:.*]] = mhlo.constant dense<4.000000e+00>
  // CHECK: %[[TMP_2:.*]] = mhlo.clamp %[[TMP_0]], %[[ARG]], %[[TMP_1]]
  // CHECK: %[[TMP_3:.*]] = mhlo.multiply %[[TMP_2]], %[[TMP_2]]
  // CHECK: %[[TMP_6:.*]] = mhlo.constant dense<-2.72614237E-10>
  // CHECK: %[[TMP_8:.*]] = mhlo.multiply %[[TMP_6]], %[[TMP_3]]
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
  // CHECK: %[[TMP_28:.*]] = mhlo.constant dense<-1.45660715E-5>
  // CHECK: %[[TMP_30:.*]] = mhlo.multiply %[[TMP_28]], %[[TMP_3]]
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
  // CHECK: %[[TMP_43:.*]] = mhlo.divide %[[TMP_42]], %[[TMP_41]]
  // CHECK-DAG: %[[TMP_44:.*]] = mhlo.constant dense<-1.000000e+00>
  // CHECK-DAG: %[[TMP_45:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[RESULT:.*]] = mhlo.clamp %[[TMP_44]], %[[TMP_43]], %[[TMP_45]]
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @erf_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func.func @erf_f16(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: mhlo.convert %[[ARG]] : (tensor<f16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = mhlo.convert %{{.*}} : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @erf_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func.func @erf_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  // CHECK: mhlo.convert %[[ARG]] : (tensor<bf16>) -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = mhlo.convert %{{.*}} : (tensor<f32>) -> tensor<bf16>
  // CHECK: return %[[RESULT]]
  %1 = "chlo.erf"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %1 : tensor<bf16>
}


// CHECK-LABEL: @top_k
// CHECK-SAME: (%[[ARG:.*]]: tensor<16x16xf32>)
func.func @top_k(%arg : tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>) {
  // CHECK:      %[[IOTA:.*]] = "mhlo.iota"() {iota_dimension = 1 : i64}
  // CHECK-NEXT: %[[SORT:.*]]:2 = "mhlo.sort"(%[[ARG]], %[[IOTA]]) ({
  // CHECK-NEXT: ^{{.*}}(%[[LHS:.*]]: tensor<f32>, %[[RHS:.*]]: tensor<f32>, %{{.*}}: tensor<i32>, %{{.*}}: tensor<i32>):
  // CHECK-NEXT:   %[[CMP:.*]] = mhlo.compare GT, %[[LHS]], %[[RHS]], TOTALORDER
  // CHECK-NEXT:   mhlo.return %[[CMP]]
  // CHECK-NEXT: }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  // CHECK-NEXT: %[[VAL:.*]] = "mhlo.slice"(%[[SORT]]#0) {limit_indices = dense<[16, 8]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-NEXT: %[[IDX:.*]] = "mhlo.slice"(%[[SORT]]#1) {limit_indices = dense<[16, 8]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-NEXT: return %[[VAL]], %[[IDX]]
  %1:2 = chlo.top_k(%arg, k=8) : tensor<16x16xf32> -> (tensor<16x8xf32>, tensor<16x8xi32>)
  func.return %1#0, %1#1 : tensor<16x8xf32>, tensor<16x8xi32>
}

// -----

// CHECK-LABEL: @dyn_top_k
// CHECK-SAME: ([[ARG:%.*]]: tensor<?x5x?xi1>
// CHECK-SAME: -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>)
func.func @dyn_top_k(%arg0: tensor<?x5x?xi1>) -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>) {
  // CHECK-NEXT: [[DIM_0_I32:%.*]] = "mhlo.get_dimension_size"([[ARG]]) {dimension = 0 : i64} : (tensor<?x5x?xi1>) -> tensor<i32>
  // CHECK-NEXT: [[DIM_0_I32x1:%.*]] = mhlo.reshape [[DIM_0_I32]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: [[DIM_1_I32:%.*]] = "mhlo.get_dimension_size"([[ARG]]) {dimension = 1 : i64} : (tensor<?x5x?xi1>) -> tensor<i32>
  // CHECK-NEXT: [[DIM_1_I32x1:%.*]] = mhlo.reshape [[DIM_1_I32]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: [[DIM_2_I32:%.*]] = "mhlo.get_dimension_size"([[ARG]]) {dimension = 2 : i64} : (tensor<?x5x?xi1>) -> tensor<i32>
  // CHECK-NEXT: [[DIM_2_I32x1:%.*]] = mhlo.reshape [[DIM_2_I32]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: [[IOTA_SHAPE:%.*]] = "mhlo.concatenate"([[DIM_0_I32x1]], [[DIM_1_I32x1]], [[DIM_2_I32x1]]) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[K_I32:%.*]] = mhlo.constant dense<2> : tensor<i32>
  // CHECK-NEXT: [[K_I32x1:%.*]] = mhlo.reshape [[K_I32]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: [[RESULT_SHAPE:%.*]] = "mhlo.concatenate"([[DIM_0_I32x1]], [[DIM_1_I32x1]], [[K_I32x1]]) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[IOTA:%.*]] = "mhlo.dynamic_iota"([[IOTA_SHAPE]]) {iota_dimension = 2 : i64} : (tensor<3xi32>) -> tensor<?x5x?xi32>
  // CHECK-NEXT: [[SORT:%.*]]:2 = "mhlo.sort"([[ARG]], [[IOTA]]) ({
  // CHECK-NEXT: ^bb0([[ARG_1:%.*]]: tensor<i1>, [[ARG_2:%.*]]: tensor<i1>, [[ARG_3:%.*]]: tensor<i32>, [[ARG_4:%.*]]: tensor<i32>):
  // CHECK-NEXT:   [[CMP:%.*]] = mhlo.compare  GT, [[ARG_1]], [[ARG_2]],  NOTYPE : (tensor<i1>, tensor<i1>) -> tensor<i1>
  // CHECK-NEXT:   mhlo.return [[CMP]] : tensor<i1>
  // CHECK-NEXT: }) {dimension = 2 : i64, is_stable = true} : (tensor<?x5x?xi1>, tensor<?x5x?xi32>) -> (tensor<?x5x?xi1>, tensor<?x5x?xi32>)
  // CHECK-NEXT: [[STARTS:%.*]] = mhlo.constant dense<0> : tensor<3xi64>
  // CHECK-NEXT: [[LIMITS:%.*]] = mhlo.convert [[RESULT_SHAPE]] : (tensor<3xi32>) -> tensor<3xi64>
  // CHECK-NEXT: [[STRIDES:%.*]] = mhlo.constant dense<1> : tensor<3xi64>
  // CHECK-NEXT: [[VAL:%.*]] = mhlo.real_dynamic_slice [[SORT]]#0, [[STARTS]], [[LIMITS]], [[STRIDES]] : (tensor<?x5x?xi1>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x5x2xi1>
  // CHECK-NEXT: [[IDX:%.*]] = mhlo.real_dynamic_slice [[SORT]]#1, [[STARTS]], [[LIMITS]], [[STRIDES]] : (tensor<?x5x?xi32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x5x2xi32>
  // CHECK-NEXT: return [[VAL]], [[IDX]] : tensor<?x5x2xi1>, tensor<?x5x2xi32>
  %values, %indices = chlo.top_k(%arg0, k = 2) : tensor<?x5x?xi1> -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>)
  return %values, %indices : tensor<?x5x2xi1>, tensor<?x5x2xi32>
}
