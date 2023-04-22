// RUN: mlir-hlo-opt --mhlo-legalize-trigonometric-to-approximation --split-input-file %s | FileCheck %s

// CHECK-LABEL: @tanh_f64
func @tanh_f64(%arg0 : f64) -> f64 {
  // CHECK: tanh
  %res = math.tanh %arg0 : f64
  return %res : f64
}

// -----

// CHECK-LABEL: @tanh_f32
// CHECK-SAME: (%[[ARG:.*]]: f32) -> f32
func @tanh_f32(%arg0 : f32) -> f32 {
  // CHECK-DAG: %[[C:.*]] = constant -2.76076837E-16 : f32
  // CHECK-DAG: %[[C0:.*]] = constant 2.00018794E-13 : f32
  // CHECK-DAG: %[[C1:.*]] = constant -8.60467184E-11 : f32
  // CHECK-DAG: %[[C2:.*]] = constant 5.12229725E-8 : f32
  // CHECK-DAG: %[[C3:.*]] = constant 1.48572235E-5 : f32
  // CHECK-DAG: %[[C4:.*]] = constant 6.37261954E-4 : f32
  // CHECK-DAG: %[[C5:.*]] = constant 0.00489352457 : f32
  // CHECK-DAG: %[[C6:.*]] = constant 1.19825836E-6 : f32
  // CHECK-DAG: %[[C7:.*]] = constant 1.18534706E-4 : f32
  // CHECK-DAG: %[[C8:.*]] = constant 0.00226843474 : f32
  // CHECK-DAG: %[[C9:.*]] = constant 0.00489352504 : f32
  // CHECK-DAG: %[[C10:.*]] = constant 4.000000e-04 : f32
  // CHECK-DAG: %[[C11:.*]] = constant 7.90531111 : f32
  // CHECK-DAG: %[[C12:.*]] = constant -7.90531111 : f32
  // CHECK-DAG: %[[C13:.*]] = constant 1.000000e+00 : f32
  // CHECK-DAG: %[[C14:.*]] = constant -1.000000e+00 : f32
  // CHECK-DAG: %[[TMP0:.*]] = mulf %[[ARG]], %[[ARG]] : f32
  // CHECK-DAG: %[[TMP1:.*]] = mulf %[[TMP0]], %[[C]] : f32
  // CHECK-DAG: %[[TMP2:.*]] = addf %[[TMP1]], %[[C0]] : f32
  // CHECK-DAG: %[[TMP3:.*]] = mulf %[[TMP0]], %[[TMP2]] : f32
  // CHECK-DAG: %[[TMP4:.*]] = addf %[[TMP3]], %[[C1]] : f32
  // CHECK-DAG: %[[TMP5:.*]] = mulf %[[TMP0]], %[[TMP4]] : f32
  // CHECK-DAG: %[[TMP6:.*]] = addf %[[TMP5]], %[[C2]] : f32
  // CHECK-DAG: %[[TMP7:.*]] = mulf %[[TMP0]], %[[TMP6]] : f32
  // CHECK-DAG: %[[TMP8:.*]] = addf %[[TMP7]], %[[C3]] : f32
  // CHECK-DAG: %[[TMP9:.*]] = mulf %[[TMP0]], %[[TMP8]] : f32
  // CHECK-DAG: %[[TMP10:.*]] = addf %[[TMP9]], %[[C4]] : f32
  // CHECK-DAG: %[[TMP11:.*]] = mulf %[[TMP0]], %[[TMP10]] : f32
  // CHECK-DAG: %[[TMP12:.*]] = addf %[[TMP11]], %[[C5]] : f32
  // CHECK-DAG: %[[TMP13:.*]] = mulf %[[ARG]], %[[TMP12]] : f32
  // CHECK-DAG: %[[TMP14:.*]] = mulf %[[TMP0]], %[[C6]] : f32
  // CHECK-DAG: %[[TMP15:.*]] = addf %[[TMP14]], %[[C7]] : f32
  // CHECK-DAG: %[[TMP16:.*]] = mulf %[[TMP0]], %[[TMP15]] : f32
  // CHECK-DAG: %[[TMP17:.*]] = addf %[[TMP16]], %[[C8]] : f32
  // CHECK-DAG: %[[TMP18:.*]] = mulf %[[TMP0]], %[[TMP17]] : f32
  // CHECK-DAG: %[[TMP19:.*]] = addf %[[TMP18]], %[[C9]] : f32
  // CHECK-DAG: %[[TMP20:.*]] = divf %[[TMP13]], %[[TMP19]] : f32
  // CHECK-DAG: %[[TMP21:.*]] = absf %[[ARG]] : f32
  // CHECK-DAG: %[[TMP22:.*]] = cmpf olt, %[[TMP21]], %[[C10]] : f32
  // CHECK-DAG: %[[TMP23:.*]] = select %[[TMP22]], %[[ARG]], %[[TMP20]] : f32
  // CHECK-DAG: %[[TMP24:.*]] = cmpf ugt, %[[ARG]], %[[C11]] : f32
  // CHECK-DAG: %[[TMP25:.*]] = cmpf ult, %[[ARG]], %[[C12]] : f32
  // CHECK-DAG: %[[IS_NAN:.*]] = cmpf une, %[[ARG]], %[[ARG]] : f32
  // CHECK-DAG: %[[TMP26:.*]] = select %[[TMP24]], %[[C13]], %[[TMP23]] : f32
  // CHECK-DAG: %[[TMP27:.*]] = select %[[TMP25]], %[[C14]], %[[TMP26]] : f32
  // CHECK-DAG: %[[RESULT:.*]] = select %[[IS_NAN]], %[[ARG]], %[[TMP27]] : f32
  // CHECK: return %[[RESULT]] : f32
  %res = math.tanh %arg0 : f32
  return %res : f32
}

// -----

func @tanh_f16(%arg0 : f16) -> f16 {
  // CHECK-LABEL: func @tanh_f16
  // CHECK-SAME: (%[[ARG:.*]]: f16) -> f16
  // CHECK: %{{.*}} = fpext %[[ARG]] : f16 to f32
  // CHECK: %[[RES:.*]] = fptrunc %{{.*}} : f32 to f16
  // CHECK: return %[[RES]] : f16
  %res = math.tanh %arg0 : f16
  return %res : f16
}

// -----

// CHECK-LABEL: @atan2_f64
func @atan2_f64(%arg0 : f64, %arg1 : f64) -> f64 {
  // CHECK: atan2
  %res = math.atan2 %arg0, %arg1 : f64
  return %res : f64
}

// -----

// CHECK-LABEL: @atan_f64
func @atan_f64(%arg : f64) -> f64 {
  // CHECK: atan
  %res = math.atan %arg : f64
  return %res : f64
}
