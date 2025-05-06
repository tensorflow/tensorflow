// RUN: tf-quant-opt %s -tf-quant-optimize -allow-unregistered-dialect | FileCheck %s

func.func @remove_redundant_cast(%arg0: tensor<1x100x100x1xf32>) -> (tensor<1x96x96x1xf32>) {
  %cst = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
  %cst_0 = "tf.Const"() {value = dense<0.0235294122> : tensor<f32>} : () -> tensor<f32>
  %cst_1 = "tf.Const"() {value = dense<0.00708661414> : tensor<1xf32>} : () -> tensor<1xf32>
  %cst_2 = "tf.Const"() {value = dense<1.799000e+03> : tensor<1xf32>} : () -> tensor<1xf32>
  %cst_3 = "tf.Const"() {value = dense<[[[[1.400000e+01]], [[-2.800000e+01]], [[4.200000e+01]]], [[[-5.600000e+01]], [[7.100000e+01]], [[-8.500000e+01]]], [[[9.900000e+01]], [[-1.130000e+02]], [[1.270000e+02]]]]> : tensor<3x3x1x1xf32>} : () -> tensor<3x3x1x1xf32>
  %cst_4 = "tf.Const"() {value = dense<-1.280000e+02> : tensor<f32>} : () -> tensor<f32>
  %cst_5 = "tf.Const"() {value = dense<0.00118110236> : tensor<1xf32>} : () -> tensor<1xf32>
  %cst_6 = "tf.Const"() {value = dense<1.079500e+04> : tensor<1xf32>} : () -> tensor<1xf32>
  %cst_7 = "tf.Const"() {value = dense<0.00392156886> : tensor<f32>} : () -> tensor<f32>
  %cst_8 = "tf.Const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
  %cst_9 = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Div"(%arg0, %cst_7) : (tensor<1x100x100x1xf32>, tensor<f32>) -> tensor<1x100x100x1xf32>
  %1 = "tf.Round"(%0) : (tensor<1x100x100x1xf32>) -> tensor<1x100x100x1xf32>
  %2 = "tf.Cast"(%1) : (tensor<1x100x100x1xf32>) -> tensor<1x100x100x1xi32>
  %3 = "tf.AddV2"(%2, %cst) : (tensor<1x100x100x1xi32>, tensor<i32>) -> tensor<1x100x100x1xi32>

  %4 = "tf.ClipByValue"(%3, %cst, %cst_9) : (tensor<1x100x100x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x100x100x1xi32>
  %5 = "tf.Cast"(%4) {Truncate = false} : (tensor<1x100x100x1xi32>) -> tensor<1x100x100x1xi8>
  %6 = "tf.Cast"(%5) {Truncate = false} : (tensor<1x100x100x1xi8>) -> tensor<1x100x100x1xf32>

  %7 = "tf.Sub"(%6, %cst_4) : (tensor<1x100x100x1xf32>, tensor<f32>) -> tensor<1x100x100x1xf32>
  %8 = "tf.Conv2D"(%7, %cst_3) {dilations = [1, 1, 1, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x100x100x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x98x98x1xf32>
  %9 = "tf.AddV2"(%8, %cst_6) : (tensor<1x98x98x1xf32>, tensor<1xf32>) -> tensor<1x98x98x1xf32>
  %10 = "tf.Mul"(%9, %cst_5) : (tensor<1x98x98x1xf32>, tensor<1xf32>) -> tensor<1x98x98x1xf32>
  %11 = "tf.AddV2"(%10, %cst_8) : (tensor<1x98x98x1xf32>, tensor<f32>) -> tensor<1x98x98x1xf32>
  %12 = "tf.Floor"(%11) : (tensor<1x98x98x1xf32>) -> tensor<1x98x98x1xf32>
  %13 = "tf.Cast"(%12) {Truncate = false} : (tensor<1x98x98x1xf32>) -> tensor<1x98x98x1xi32>
  %14 = "tf.AddV2"(%13, %cst) : (tensor<1x98x98x1xi32>, tensor<i32>) -> tensor<1x98x98x1xi32>

  %15 = "tf.ClipByValue"(%14, %cst, %cst_9) : (tensor<1x98x98x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x98x98x1xi32>
  %16 = "tf.Cast"(%15) {Truncate = false} : (tensor<1x98x98x1xi32>) -> tensor<1x98x98x1xi8>
  %17 = "tf.Cast"(%16) {Truncate = false} : (tensor<1x98x98x1xi8>) -> tensor<1x98x98x1xf32>

  %18 = "tf.Sub"(%17, %cst_4) : (tensor<1x98x98x1xf32>, tensor<f32>) -> tensor<1x98x98x1xf32>
  %19 = "tf.Conv2D"(%18, %cst_3) {dilations = [1, 1, 1, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x98x98x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x96x96x1xf32>
  %20 = "tf.AddV2"(%19, %cst_2) : (tensor<1x96x96x1xf32>, tensor<1xf32>) -> tensor<1x96x96x1xf32>
  %21 = "tf.Mul"(%20, %cst_1) : (tensor<1x96x96x1xf32>, tensor<1xf32>) -> tensor<1x96x96x1xf32>
  %22 = "tf.AddV2"(%21, %cst_8) : (tensor<1x96x96x1xf32>, tensor<f32>) -> tensor<1x96x96x1xf32>
  %23 = "tf.Floor"(%22) : (tensor<1x96x96x1xf32>) -> tensor<1x96x96x1xf32>
  %24 = "tf.Cast"(%23) {Truncate = false} : (tensor<1x96x96x1xf32>) -> tensor<1x96x96x1xi32>
  %25 = "tf.AddV2"(%24, %cst) : (tensor<1x96x96x1xi32>, tensor<i32>) -> tensor<1x96x96x1xi32>

  %26 = "tf.ClipByValue"(%25, %cst, %cst_9) : (tensor<1x96x96x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x96x96x1xi32>
  %27 = "tf.Cast"(%26) {Truncate = false} : (tensor<1x96x96x1xi32>) -> tensor<1x96x96x1xi8>
  %28 = "tf.Cast"(%27) : (tensor<1x96x96x1xi8>) -> tensor<1x96x96x1xi32>

  %29 = "tf.Sub"(%28, %cst) : (tensor<1x96x96x1xi32>, tensor<i32>) -> tensor<1x96x96x1xi32>
  %30 = "tf.Cast"(%29) : (tensor<1x96x96x1xi32>) -> tensor<1x96x96x1xf32>
  %31 = "tf.Mul"(%30, %cst_0) : (tensor<1x96x96x1xf32>, tensor<f32>) -> tensor<1x96x96x1xf32>
  return %31 : tensor<1x96x96x1xf32>

// CHECK-LABEL: func.func @remove_redundant_cast

// CHECK: %[[CLIPBYVALUE_0:.*]] = "tf.ClipByValue"
// CHECK-SAME: (tensor<1x100x100x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x100x100x1xi32>
// CHECK: %[[CAST_1:.*]] = "tf.Cast"(%[[CLIPBYVALUE_0]]) <{Truncate = false}> : (tensor<1x100x100x1xi32>) -> tensor<1x100x100x1xf32>

// CHECK: %[[CLIPBYVALUE_1:.*]] = "tf.ClipByValue"
// CHECK-SAME: (tensor<1x98x98x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x98x98x1xi32>
// CHECK: %[[CAST_3:.*]] = "tf.Cast"(%[[CLIPBYVALUE_1]]) <{Truncate = false}> : (tensor<1x98x98x1xi32>) -> tensor<1x98x98x1xf32>

// CHECK: %[[CLIPBYVALUE_2:.*]] = "tf.ClipByValue"
// CHECK-SAME: (tensor<1x96x96x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x96x96x1xi32>
// CHECK: %[[SUB_2:.*]] = "tf.Sub"(%[[CLIPBYVALUE_2]], {{.*}}) : (tensor<1x96x96x1xi32>, tensor<i32>) -> tensor<1x96x96x1xi32>
}

func.func @consecutive_add_add(%arg0: tensor<i32>) -> (tensor<i32>) {
  %cst = "tf.Const"() {value = dense<-18> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {value = dense<-12> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.AddV2"(%arg0, %cst) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.AddV2"(%0, %cst_1) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>

// CHECK-LABEL: func.func @consecutive_add_add

// CHECK: %[[CST:.*]] = "tf.Const"() <{value = dense<-30> : tensor<i32>}> : () -> tensor<i32>
// CHECK: %[[ADD:.*]] = "tf.AddV2"(%arg0, %[[CST]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK: return %[[ADD]] : tensor<i32>
}

func.func @consecutive_add_sub(%arg0: tensor<i32>) -> (tensor<i32>) {
  %cst = "tf.Const"() {value = dense<-18> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {value = dense<-12> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.AddV2"(%arg0, %cst) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Sub"(%0, %cst_1) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>

// CHECK-LABEL: func.func @consecutive_add_sub

// CHECK: %[[CST:.*]] = "tf.Const"() <{value = dense<6> : tensor<i32>}> : () -> tensor<i32>
// CHECK: %[[SUB:.*]] = "tf.Sub"(%arg0, %[[CST]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK: return %[[SUB]] : tensor<i32>
}

func.func @consecutive_sub_add(%arg0: tensor<i32>) -> (tensor<i32>) {
  %cst = "tf.Const"() {value = dense<-18> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {value = dense<-12> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Sub"(%arg0, %cst) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.AddV2"(%0, %cst_1) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>

// CHECK-LABEL: func.func @consecutive_sub_add

// CHECK: %[[CST:.*]] = "tf.Const"() <{value = dense<6> : tensor<i32>}> : () -> tensor<i32>
// CHECK: %[[ADD:.*]] = "tf.AddV2"(%arg0, %[[CST]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK: return %[[ADD]] : tensor<i32>
}

func.func @consecutive_sub_sub(%arg0: tensor<i32>) -> (tensor<i32>) {
  %cst = "tf.Const"() {value = dense<-18> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {value = dense<-12> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Sub"(%arg0, %cst) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Sub"(%0, %cst_1) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>

// CHECK-LABEL: func.func @consecutive_sub_sub

// CHECK: %[[CST:.*]] = "tf.Const"() <{value = dense<-30> : tensor<i32>}> : () -> tensor<i32>
// CHECK: %[[SUB:.*]] = "tf.Sub"(%arg0, %[[CST]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK: return %[[SUB]] : tensor<i32>
}
