// RUN: tf-opt --tfl-legalize-tf-while %s -o - | FileCheck %s
// RUN: tf-opt --tfl-legalize-tf-while %s -o - --tfl-legalize-tf-while --inline='default-pipeline=''' | FileCheck %s --check-prefix=INLINE
// RUN: tf-opt --tfl-legalize-tf-while %s -o - --tfl-legalize-tf-while --inline | FileCheck %s --check-prefix=CANON

func @while_main(%arg0: tensor<?x256x256xf32>) -> (tensor<i32>, tensor<256x256xf32>, tensor<?x256x256xf32>) attributes {tf.entry_function = {inputs = "input", outputs = "Identity,Identity_1,Identity_2"}} {
  %cst = constant dense<1.000000e+00> : tensor<256x256xf32>
  %cst_0 = constant dense<0> : tensor<i32>
  %cst_1 = constant dense<-1> : tensor<i32>
  %0:5 = "tf.While"(%cst_0, %cst_1, %cst_0, %cst, %arg0) {body = @while_body_11_frozen0, cond = @while_cond_10_frozen0, device = "", is_stateless = true} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<256x256xf32>, tensor<?x256x256xf32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<256x256xf32>, tensor<?x256x256xf32>)
  return %0#2, %0#3, %0#4 : tensor<i32>, tensor<256x256xf32>, tensor<?x256x256xf32>
}

func @while_body_11_frozen0(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: tensor<*xi32>, %arg3: tensor<*xf32>, %arg4: tensor<*xf32>) -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*xf32>, tensor<*xf32>) {
  %cst = constant dense<[1, 0]> : tensor<2xi32>
  %cst_0 = constant dense<0> : tensor<i32>
  %cst_1 = constant dense<-1> : tensor<i32>
  %cst_2 = constant dense<1> : tensor<i32>
  %0 = "tf.AddV2"(%arg2, %cst_2) {T = i32, device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %1 = "tf.Transpose"(%arg3, %cst) {T = f32, Tperm = i32, device = ""} : (tensor<*xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %2 = "tf.Rank"(%arg3) : (tensor<*xf32>) -> tensor<i32>
  %3 = "tf.Range"(%2, %cst_0, %cst_1) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %4 = "tf.Sub"(%3, %cst_2) : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  %5 = "tf.Transpose"(%arg3, %4) : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
  %6 = "tf.MatMul"(%1, %5) {transpose_a = false, transpose_b = true} : (tensor<?x?xf32>, tensor<*xf32>) -> tensor<?x?xf32>
  %7 = "tf.AddV2"(%arg4, %6) {T = f32, device = ""} : (tensor<*xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  %8 = "tf.AddV2"(%arg0, %cst_2) {T = i32, device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  return %8, %arg1, %0, %arg3, %7 : tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*xf32>, tensor<*xf32>
}

func @while_cond_10_frozen0(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: tensor<*xi32>, %arg3: tensor<*xf32>, %arg4: tensor<*xf32>) -> tensor<*xi1> {
  %cst = constant dense<10> : tensor<i32>
  %0 = "tf.Less"(%arg2, %cst) {T = i32, device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>
}

// CHECK: tfl.while
// CHECK: ^bb0([[ARGS:.*]]):
// CHECK:   call @while_cond_10_frozen0
// CHECK:   yield
// CHECK: ^bb0([[ARGS]]):
// CHECK:   call @while_body
// CHECK:   yield
// CHECK: while_body
// CHECK: while_cond

// INLINE: tfl.while
// INLINE: ^bb0([[ARGS:.*]]):
// INLINE:   tf.Less
// INLINE:   yield
// INLINE: ^bb0([[ARGS]]):
// INLINE:   %cst_2 = constant
// INLINE:   yield
// INLINE-NOT: while_body
// INLINE-NOT: while_cond

// CANON-LABEL:   func @while_main
// CANON-SAME:         ([[VAL_0:%.*]]: tensor<?x256x256xf32>)
// CANON-SAME:         (tensor<i32>, tensor<256x256xf32>, tensor<?x256x256xf32>)
// CANON:           [[VAL_1:%.*]] = constant dense<1.000000e+00> : tensor<256x256xf32>
// CANON:           [[VAL_2:%.*]] = constant dense<0> : tensor<i32>
// CANON:           [[VAL_6:%.*]]:3 = "tfl.while"([[VAL_2]], [[VAL_2]], [[VAL_0]]) ( {
// CANON:           ^bb0([[VAL_7:%.*]]: tensor<*xi32>, [[VAL_8:%.*]]: tensor<*xi32>, [[VAL_9:%.*]]: tensor<*xf32>):
// CANON:             [[VAL_3:%.*]] = constant dense<10> : tensor<i32>
// CANON:             [[VAL_10:%.*]] = "tf.Less"([[VAL_8]], [[VAL_3]])
// CANON:             "tfl.yield"([[VAL_10]]) : (tensor<*xi1>) -> ()
// CANON:           },  {
// CANON:           ^bb0([[VAL_11:%.*]]: tensor<*xi32>, [[VAL_12:%.*]]: tensor<*xi32>, [[VAL_13:%.*]]: tensor<*xf32>):
// CANON:             [[VAL_4:%.*]] = constant dense<1> : tensor<i32>
// CANON:             [[VAL_5:%.*]] = "tf.Const"() {value = dense<2.560000e+02> : tensor<256x256xf32>} : () -> tensor<?x?xf32>
// CANON:             [[VAL_14:%.*]] = "tf.AddV2"([[VAL_12]], [[VAL_4]])
// CANON:             [[VAL_15:%.*]] = "tf.AddV2"([[VAL_13]], [[VAL_5]])
// CANON:             [[VAL_16:%.*]] = "tf.AddV2"([[VAL_11]], [[VAL_4]])
// CANON:             "tfl.yield"([[VAL_16]], [[VAL_14]], [[VAL_15]]) : (tensor<*xi32>, tensor<*xi32>, tensor<*xf32>) -> ()
// CANON:           }) {is_stateless = true} : (tensor<i32>, tensor<i32>, tensor<?x256x256xf32>) -> (tensor<i32>, tensor<i32>, tensor<?x256x256xf32>)
// CANON:           return [[VAL_17:%.*]]#1, [[VAL_1]], [[VAL_17]]#2 : tensor<i32>, tensor<256x256xf32>, tensor<?x256x256xf32>
// CANON:         }
