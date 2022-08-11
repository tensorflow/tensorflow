// RUN: tf-tfrt-opt %s -tf-jitrt-math-approximation="oplist=all"               \
// RUN: | FileCheck %s
// RUN: tf-tfrt-opt %s -tf-jitrt-math-approximation="oplist=exp"               \
// RUN: | FileCheck --check-prefix=EXP %s
// RUN: tf-tfrt-opt %s -tf-jitrt-math-approximation="oplist=expm1"             \
// RUN: | FileCheck --check-prefix=EXPM1 %s
// RUN: tf-tfrt-opt %s -tf-jitrt-math-approximation                            \
// RUN: | FileCheck --check-prefix=NOOP %s

// CHECK-LABEL: func @exp_scalar(
// CHECK-SAME:                   %[[VAL_0:.*]]: f32) -> f32 {
// EXP-NOT: math.exp
// NOOP: math.exp
// CHECK-DAG:  %[[CST_ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:  %[[CST_CEPHES_EXP_P5:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:  %[[CST_EXP_HI:.*]] = arith.constant 8.872300e+01 : f32
// CHECK-DAG:  %[[CST_EXP_LO:.*]] = arith.constant -8.872300e+01 : f32
// CHECK-DAG:  %[[CST_CEPHES_LOG2E:.*]] = arith.constant 1.44269502 : f32
// CHECK-DAG:  %[[CST_CEPHES_EXP_P0:.*]] = arith.constant 1.98756912E-4 : f32
// CHECK-DAG:  %[[CST_CEPHES_EXP_P1:.*]] = arith.constant 0.00139819994 : f32
// CHECK-DAG:  %[[CST_CEPHES_EXP_P2:.*]] = arith.constant 0.00833345205 : f32
// CHECK-DAG:  %[[CST_CEPHES_EXP_P3:.*]] = arith.constant 0.0416657962 : f32
// CHECK-DAG:  %[[CST_CEPHES_EXP_P4:.*]] = arith.constant 0.166666657 : f32
// CHECK-DAG:  %[[CST_CEPHES_EXP_C1:.*]] = arith.constant -0.693359375 : f32
// CHECK-DAG:  %[[CST_CEPHES_EXP_C2:.*]] = arith.constant 2.12194442E-4 : f32
// CHECK-DAG:  %[[CST_MANT_BITS:.*]] = arith.constant 23 : i32
// CHECK-DAG:  %[[CST_MAX_EXPONENT:.*]] = arith.constant 2.780000e+02 : f32
// CHECK-DAG:  %[[CST_MIN_EXPONENT:.*]] = arith.constant -2.780000e+02 : f32
// CHECK-DAG:  %[[CST_BIAS:.*]] = arith.constant 127 : i32
// CHECK-DAG:  %[[CST_TWO:.*]] = arith.constant 2 : i32
// CHECK:  %[[VAL_18:.*]] = arith.cmpf olt, %[[VAL_0]], %[[CST_EXP_HI]] : f32
// CHECK:  %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_0]], %[[CST_EXP_HI]] : f32
// CHECK:  %[[VAL_20:.*]] = arith.cmpf ogt, %[[VAL_19]], %[[CST_EXP_LO]] : f32
// CHECK:  %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_19]], %[[CST_EXP_LO]] : f32
// CHECK:  %[[VAL_22:.*]] = math.fma %[[VAL_21]], %[[CST_CEPHES_LOG2E]], %[[CST_CEPHES_EXP_P5]] : f32
// CHECK:  %[[VAL_23:.*]] = math.floor %[[VAL_22]] : f32
// CHECK:  %[[VAL_24:.*]] = math.fma %[[VAL_23]], %[[CST_CEPHES_EXP_C1]], %[[VAL_21]] : f32
// CHECK:  %[[VAL_25:.*]] = math.fma %[[VAL_23]], %[[CST_CEPHES_EXP_C2]], %[[VAL_24]] : f32
// CHECK:  %[[VAL_26:.*]] = arith.mulf %[[VAL_25]], %[[VAL_25]] : f32
// CHECK:  %[[VAL_27:.*]] = arith.mulf %[[VAL_26]], %[[VAL_25]] : f32
// CHECK:  %[[VAL_28:.*]] = math.fma %[[CST_CEPHES_EXP_P0]], %[[VAL_25]], %[[CST_CEPHES_EXP_P1]] : f32
// CHECK:  %[[VAL_29:.*]] = math.fma %[[CST_CEPHES_EXP_P3]], %[[VAL_25]], %[[CST_CEPHES_EXP_P4]] : f32
// CHECK:  %[[VAL_30:.*]] = arith.addf %[[VAL_25]], %[[CST_ONE]] : f32
// CHECK:  %[[VAL_31:.*]] = math.fma %[[VAL_28]], %[[VAL_25]], %[[CST_CEPHES_EXP_P2]] : f32
// CHECK:  %[[VAL_32:.*]] = math.fma %[[VAL_29]], %[[VAL_25]], %[[CST_CEPHES_EXP_P5]] : f32
// CHECK:  %[[VAL_33:.*]] = math.fma %[[VAL_31]], %[[VAL_27]], %[[VAL_32]] : f32
// CHECK:  %[[VAL_34:.*]] = math.fma %[[VAL_33]], %[[VAL_26]], %[[VAL_30]] : f32
// CHECK:  %[[VAL_35:.*]] = arith.cmpf olt, %[[VAL_23]], %[[CST_MAX_EXPONENT]] : f32
// CHECK:  %[[VAL_36:.*]] = arith.select %[[VAL_35]], %[[VAL_23]], %[[CST_MAX_EXPONENT]] : f32
// CHECK:  %[[VAL_37:.*]] = arith.cmpf ogt, %[[VAL_36]], %[[CST_MIN_EXPONENT]] : f32
// CHECK:  %[[VAL_38:.*]] = arith.select %[[VAL_37]], %[[VAL_36]], %[[CST_MIN_EXPONENT]] : f32
// CHECK:  %[[VAL_39:.*]] = arith.fptosi %[[VAL_38]] : f32 to i32
// CHECK:  %[[VAL_40:.*]] = arith.shrsi %[[VAL_39]], %[[CST_TWO]] : i32
// CHECK:  %[[VAL_41:.*]] = arith.addi %[[VAL_40]], %[[CST_BIAS]] : i32
// CHECK:  %[[VAL_42:.*]] = arith.shli %[[VAL_41]], %[[CST_MANT_BITS]] : i32
// CHECK:  %[[VAL_43:.*]] = arith.bitcast %[[VAL_42]] : i32 to f32
// CHECK:  %[[VAL_44:.*]] = arith.mulf %[[VAL_34]], %[[VAL_43]] : f32
// CHECK:  %[[VAL_45:.*]] = arith.mulf %[[VAL_44]], %[[VAL_43]] : f32
// CHECK:  %[[VAL_46:.*]] = arith.mulf %[[VAL_45]], %[[VAL_43]] : f32
// CHECK:  %[[VAL_47:.*]] = arith.subi %[[VAL_39]], %[[VAL_40]] : i32
// CHECK:  %[[VAL_48:.*]] = arith.subi %[[VAL_47]], %[[VAL_40]] : i32
// CHECK:  %[[VAL_49:.*]] = arith.subi %[[VAL_48]], %[[VAL_40]] : i32
// CHECK:  %[[VAL_50:.*]] = arith.addi %[[VAL_49]], %[[CST_BIAS]] : i32
// CHECK:  %[[VAL_51:.*]] = arith.shli %[[VAL_50]], %[[CST_MANT_BITS]] : i32
// CHECK:  %[[VAL_52:.*]] = arith.bitcast %[[VAL_51]] : i32 to f32
// CHECK:  %[[VAL_53:.*]] = arith.mulf %[[VAL_46]], %[[VAL_52]] : f32
// CHECK:  %[[VAL_54:.*]] = arith.cmpf ogt, %[[VAL_53]], %[[VAL_0]] : f32
// CHECK:  %[[VAL_55:.*]] = arith.select %[[VAL_54]], %[[VAL_53]], %[[VAL_0]] : f32
// CHECK:  return %[[VAL_55]] : f32
// CHECK: }
func.func @exp_scalar(%arg0: f32) -> f32 {
  %0 = math.exp %arg0 : f32
  func.return %0 : f32
}

// CHECK-LABEL: func @expm1_scalar(
// CHECK-NOT: math.exp
// EXP: math.expm1
// EXPM1-NOT: math.expm1
// NOOP: math.expm1
// CHECK:    %[[VAL_38:.*]] = arith.cmpf ueq, %[[VAL_37:.*]], %cst
// CHECK:    %[[VAL_39:.*]] = arith.subf %[[VAL_37]], %cst
// CHECK:    %[[VAL_40:.*]] = arith.cmpf oeq, %[[VAL_39]], %cst_0
// CHECK:    %[[VAL_41:.*]] = math.log %[[VAL_37]]
// CHECK:    %[[VAL_42:.*]] = arith.cmpf oeq, %[[VAL_41]], %[[VAL_37]]
// CHECK:    %[[VAL_43:.*]] = arith.divf %arg0, %[[VAL_41]]
// CHECK:    %[[VAL_44:.*]] = arith.mulf %[[VAL_39]], %[[VAL_43]]
// CHECK:    %[[VAL_45:.*]] = arith.select %[[VAL_42]], %[[VAL_37]], %[[VAL_44]]
// CHECK:    %[[VAL_46:.*]] = arith.select %[[VAL_40]], %cst_0, %[[VAL_45]] 
// CHECK:    %[[VAL_47:.*]] = arith.select %[[VAL_38]], %arg0, %[[VAL_46]] 
// CHECK: }
func.func @expm1_scalar(%arg0 : f32) -> f32 {
  %0 = math.expm1 %arg0: f32
  func.return %0 : f32
}
