// RUN: tf-opt %s -tf-functional-control-flow-to-cfg -split-input-file | FileCheck %s

func private @testIf1Then(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
func private @testIf1Else(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

// CHECK-LABEL: func @testIf1Result(%arg0: tensor<i1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>)
func @testIf1Result(tensor<i1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>):
  %1 = "tf.If"(%arg0, %arg1, %arg2) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<i1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

// CHECK:   [[TOBOOL:%.+]] = "tf.ToBool"(%arg0) : (tensor<i1>) -> tensor<i1>
// CHECK:   [[PRED:%.+]] = tensor.extract [[TOBOOL]][] : tensor<i1>
// CHECK:   cond_br [[PRED]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   [[THEN:%.+]] = call @testIf1Then(%arg1, %arg2)
// CHECK:   br ^bb3([[THEN]] : tensor<*xf32>)
// CHECK: ^bb2:
// CHECK:   [[ELSE:%.+]] = call @testIf1Else(%arg1, %arg2)
// CHECK:   br ^bb3([[ELSE]] : tensor<*xf32>)
// CHECK: ^bb3([[BBARG0:%.+]]: tensor<*xf32>):

  return %1 : tensor<*xf32>
// CHECK:   return [[BBARG0]] : tensor<*xf32>
}

func private @testIf3Then(tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>)
func private @testIf3Else(tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>)

// CHECK-LABEL: func @testIf3Result(%arg0: tensor<i1>, %arg1: tensor<*xf32>)
func @testIf3Result(tensor<i1>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>) {
^bb0(%arg0: tensor<i1>, %arg1: tensor<*xf32>):
  %1:3 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIf3Then, else_branch = @testIf3Else, is_stateless = false
  } : (tensor<i1>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>)

// CHECK:   [[TOBOOL:%.+]] = "tf.ToBool"(%arg0) : (tensor<i1>) -> tensor<i1>
// CHECK:   [[PRED:%.+]] = tensor.extract [[TOBOOL]][] : tensor<i1>
// CHECK:   cond_br [[PRED]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   [[THEN:%.+]]:3 = call @testIf3Then(%arg1)
// CHECK:   br ^bb3([[THEN]]#0, [[THEN]]#1, [[THEN]]#2 : tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>)
// CHECK: ^bb2:
// CHECK:   [[ELSE:%.+]]:3 = call @testIf3Else(%arg1)
// CHECK:   br ^bb3([[ELSE]]#0, [[ELSE]]#1, [[ELSE]]#2 : tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>)
// CHECK: ^bb3([[BBARG0:%.+]]: tensor<*xf32>, [[BBARG1:%.+]]: tensor<*xi8>, [[BBARG2:%.+]]: tensor<*xbf16>):
  return %1#0, %1#1, %1#2 : tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>
// CHECK:  return [[BBARG0]], [[BBARG1]], [[BBARG2]]
}

// -----

func @testIfThen(%arg0: tensor<!tf_type.variant>) -> tensor<!tf_type.variant> {
  return %arg0 : tensor<!tf_type.variant>
}
func @testIfElse(%arg0: tensor<!tf_type.variant>) -> tensor<!tf_type.variant> {
  return %arg0 : tensor<!tf_type.variant>
}

// CHECK-LABEL: func @testIfCasts(%arg0: tensor<i1>, %arg1: tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>>
func @testIfCasts(%arg0: tensor<i1>, %arg1: tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>> {
  %0 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen, else_branch = @testIfElse, is_stateless = false
  } : (tensor<i1>, tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>>
  return %0: tensor<!tf_type.variant<tensor<f32>>>
// CHECK:   [[TOBOOL:%.+]] = "tf.ToBool"(%arg0) : (tensor<i1>) -> tensor<i1>
// CHECK:   [[PRED:%.+]] = tensor.extract [[TOBOOL]][] : tensor<i1>
// CHECK:   cond_br [[PRED]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   [[CAST0:%.+]] = "tf.Cast"(%arg1) {Truncate = false} : (tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant>
// CHECK:   [[THEN:%.+]] = call @testIfThen([[CAST0]]) : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
// CHECK:   [[CAST1:%.+]] = "tf.Cast"([[THEN]]) {Truncate = false} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant<tensor<f32>>>
// CHECK:   br ^bb3([[CAST1]] : tensor<!tf_type.variant<tensor<f32>>>)
// CHECK: ^bb2:
// CHECK:   [[CAST2:%.+]] = "tf.Cast"(%arg1) {Truncate = false} : (tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant>
// CHECK:   [[ELSE:%.+]] = call @testIfElse([[CAST2]]) : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
// CHECK:   [[CAST3:%.+]] = "tf.Cast"([[ELSE]]) {Truncate = false} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant<tensor<f32>>>
// CHECK:   br ^bb3([[CAST3]] : tensor<!tf_type.variant<tensor<f32>>>)
// CHECK: ^bb3([[BBARG0:%.+]]: tensor<!tf_type.variant<tensor<f32>>>):
// CHECK:   return [[BBARG0]] : tensor<!tf_type.variant<tensor<f32>>>
}

// -----

// If with a 4xi1 condition.

func private @testIf1Then(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
func private @testIf1Else(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

// CHECK-LABEL: func @testIf1x4
func @testIf1x4(tensor<4xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32> {
^bb0(%arg0: tensor<4xi1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>):

  // CHECK: [[TOBOOL:%.+]] = "tf.ToBool"(%arg0) : (tensor<4xi1>) -> tensor<i1>
  // CHECK: [[PRED:%.+]] = tensor.extract [[TOBOOL]][] : tensor<i1>
  %1 = "tf.If"(%arg0, %arg1, %arg2) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<4xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  return %1 : tensor<*xf32>
}

// -----


func private @testWhile2Cond(tensor<*xf32>, tensor<*xf32>) -> (tensor<i1>)
func private @testWhile2Body(tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)

// CHECK-LABEL: func @testWhile2Result(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>)
func @testWhile2Result(tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>):
  %1:2 = "tf.While"(%arg0, %arg1) {
    cond = @testWhile2Cond, body = @testWhile2Body, is_stateless = false
  } : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)

// CHECK:   br ^bb1(%arg0, %arg1 : tensor<*xf32>, tensor<*xf32>)
// CHECK: ^bb1([[CONDARG0:%.+]]: tensor<*xf32>, [[CONDARG1:%.+]]: tensor<*xf32>):
// CHECK:   [[CONTINUE:%.+]] = call @testWhile2Cond(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<i1>
// CHECK:   [[TOBOOL:%.+]] = "tf.ToBool"([[CONTINUE]]) : (tensor<i1>) -> tensor<i1>
// CHECK:   [[PRED:%.+]] = tensor.extract [[TOBOOL]][] : tensor<i1>
// CHECK:   cond_br [[PRED]], ^bb2([[CONDARG0]], [[CONDARG1]] : tensor<*xf32>, tensor<*xf32>), ^bb3([[CONDARG0]], [[CONDARG1]] : tensor<*xf32>, tensor<*xf32>)
// CHECK: ^bb2([[BODYARG0:%.+]]: tensor<*xf32>, [[BODYARG1:%.+]]: tensor<*xf32>):
// CHECK:   [[BODYRETS:%.+]]:2 = call @testWhile2Body([[BODYARG0]], [[BODYARG1]]) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
// CHECK:   br ^bb1([[BODYRETS]]#0, [[BODYRETS]]#1 : tensor<*xf32>, tensor<*xf32>)
// CHECK: ^bb3([[EXITARG0:%.+]]: tensor<*xf32>, [[EXITARG1:%.+]]: tensor<*xf32>):

  return %1#0, %1#1 : tensor<*xf32>, tensor<*xf32>
// CHECK:   return [[EXITARG0]], [[EXITARG1]] : tensor<*xf32>, tensor<*xf32>
}


func private @testWhile0Cond() -> (tensor<i1>)
func private @testWhile0Body() -> ()

// CHECK-LABEL: func @testWhile0Result() {
func @testWhile0Result() {

^bb0:
  "tf.While"() { cond = @testWhile0Cond, body = @testWhile0Body, is_stateless = false } : () -> ()
// CHECK:   br ^bb1
// CHECK: ^bb1:
// CHECK:   [[CONTINUE:%.+]] = call @testWhile0Cond() : () -> tensor<i1>
// CHECK:   [[TOBOOL:%.+]] = "tf.ToBool"([[CONTINUE]]) : (tensor<i1>) -> tensor<i1>
// CHECK:   [[PRED:%.+]] = tensor.extract [[TOBOOL]][] : tensor<i1>
// CHECK:   cond_br [[PRED]], ^bb2, ^bb3
// CHECK: ^bb2:
// CHECK:   call @testWhile0Body() : () -> ()
// CHECK:   br ^bb1
// CHECK: ^bb3:

  return
// CHECK:   return
}


// CHECK-LABEL:  func @testComplexWhile1Result(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
func @testComplexWhile1Result(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> (tensor<*xf32>) {
  %0 = arith.addf %arg0, %arg1 : tensor<*xf32>
// CHECK:    [[ADDF:%.+]] = arith.addf %arg0, %arg1 : tensor<*xf32>

  %1:2 = "tf.While"(%arg0, %0) {
    cond = @testWhile2Cond, body = @testWhile2Body, is_stateless = false
  } : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
// CHECK:    br ^bb1(%arg0, [[ADDF]] : tensor<*xf32>, tensor<*xf32>)
// CHECK:  ^bb1([[CONDARG0:%.+]]: tensor<*xf32>, [[CONDARG1:%.+]]: tensor<*xf32>):
// CHECK:    [[CONTINUE:%.+]] = call @testWhile2Cond([[CONDARG0]], [[CONDARG1]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<i1>
// CHECK:    [[TOBOOL:%.+]] = "tf.ToBool"([[CONTINUE]]) : (tensor<i1>) -> tensor<i1>
// CHECK:    [[PRED:%.+]] = tensor.extract [[TOBOOL]][] : tensor<i1>
// CHECK:    cond_br [[PRED]], ^bb2([[CONDARG0]], [[CONDARG1]] : tensor<*xf32>, tensor<*xf32>), ^bb3([[CONDARG0]], [[CONDARG1]] : tensor<*xf32>, tensor<*xf32>)
// CHECK:  ^bb2([[BODYARG0:%.+]]: tensor<*xf32>, [[BODYARG1:%.+]]: tensor<*xf32>):
// CHECK:    [[BODYRETS:%.+]]:2 = call @testWhile2Body([[BODYARG0]], [[BODYARG1]]) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
// CHECK:    br ^bb1([[BODYRETS]]#0, [[BODYRETS]]#1 : tensor<*xf32>, tensor<*xf32>)

  br ^bb1(%1#1, %0 : tensor<*xf32>, tensor<*xf32>)
// CHECK:  ^bb3([[EXITARG0:%.+]]: tensor<*xf32>, [[EXITARG1:%.+]]: tensor<*xf32>):
// CHECK:    br ^bb4([[EXITARG1]], [[ADDF]] : tensor<*xf32>, tensor<*xf32>)

^bb1(%2: tensor<*xf32>, %3: tensor<*xf32>):
  %4 = arith.subf %arg0, %arg1 : tensor<*xf32>
  return %4 : tensor<*xf32>
// CHECK:  ^bb4([[FINALARG0:%.+]]: tensor<*xf32>, [[FINALARG1:%.+]]: tensor<*xf32>):
// CHECK:    [[SUBF:%.+]] = arith.subf %arg0, %arg1 : tensor<*xf32>
// CHECK:    return [[SUBF]] : tensor<*xf32>
}

// -----

func @testWhileCond(%arg0: tensor<!tf_type.variant>) -> (tensor<i1>) {
  %true = "tf.Const"() { value = dense<true> : tensor<i1> } : () -> (tensor<i1>)
  return %true : tensor<i1>
}
func @testWhileBody(%arg0: tensor<!tf_type.variant<tensor<1x?xf32>>>) -> (tensor<!tf_type.variant<tensor<?x?xf32>>>) {
  %0 = "tf.Cast"(%arg0) : (tensor<!tf_type.variant<tensor<1x?xf32>>>) -> tensor<!tf_type.variant<tensor<?x?xf32>>>
  return %0 : tensor<!tf_type.variant<tensor<?x?xf32>>>
}

// CHECK-LABEL: func @testWhileCasts(%arg0: tensor<!tf_type.variant<tensor<1x3xf32>>>) -> tensor<!tf_type.variant<tensor<*xf32>>>
func @testWhileCasts(%arg0: tensor<!tf_type.variant<tensor<1x3xf32>>>) -> (tensor<!tf_type.variant<tensor<*xf32>>>) {
  %0 = "tf.While"(%arg0) {
    cond = @testWhileCond, body = @testWhileBody, is_stateless = false
  } : (tensor<!tf_type.variant<tensor<1x3xf32>>>) -> (tensor<!tf_type.variant<tensor<*xf32>>>)
  return %0 : tensor<!tf_type.variant<tensor<*xf32>>>
// CHECK:   [[CASTENTRY:%.+]] = "tf.Cast"(%arg0) {Truncate = false} : (tensor<!tf_type.variant<tensor<1x3xf32>>>) -> tensor<!tf_type.variant>
// CHECK:   br ^bb1([[CASTENTRY]] : tensor<!tf_type.variant>)
// CHECK: ^bb1([[CONDARG0:%.+]]: tensor<!tf_type.variant>):        // 2 preds: ^bb0, ^bb2
// CHECK:   [[CONTINUE:%.+]] = call @testWhileCond([[CONDARG0]]) : (tensor<!tf_type.variant>) -> tensor<i1>
// CHECK:   [[TOBOOL:%.+]] = "tf.ToBool"([[CONTINUE]]) : (tensor<i1>) -> tensor<i1>
// CHECK:   [[PRED:%.+]] = tensor.extract [[TOBOOL]][] : tensor<i1>
// CHECK:   [[CASTCONDARG0:%.+]] = "tf.Cast"([[CONDARG0]]) {Truncate = false} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant<tensor<1x?xf32>>>
// CHECK:   cond_br [[PRED]], ^bb2([[CASTCONDARG0]] : tensor<!tf_type.variant<tensor<1x?xf32>>>), ^bb3([[CASTCONDARG0]] : tensor<!tf_type.variant<tensor<1x?xf32>>>)
// CHECK: ^bb2([[BODYARG0:%.+]]: tensor<!tf_type.variant<tensor<1x?xf32>>>):       // pred: ^bb1
// CHECK:   [[WHILERET:%.+]] = call @testWhileBody([[BODYARG0]]) : (tensor<!tf_type.variant<tensor<1x?xf32>>>) -> tensor<!tf_type.variant<tensor<?x?xf32>>>
// CHECK:   [[CASTWHILERET:%.+]] = "tf.Cast"([[WHILERET]]) {Truncate = false} : (tensor<!tf_type.variant<tensor<?x?xf32>>>) -> tensor<!tf_type.variant>
// CHECK:   br ^bb1([[CASTWHILERET]] : tensor<!tf_type.variant>)
// CHECK: ^bb3([[EXITARG0:%.+]]: tensor<!tf_type.variant<tensor<1x?xf32>>>):       // pred: ^bb1
// CHECK:   [[CASTEXITARG0:%.+]] = "tf.Cast"([[EXITARG0]]) {Truncate = false} : (tensor<!tf_type.variant<tensor<1x?xf32>>>) -> tensor<!tf_type.variant<tensor<*xf32>>>
// CHECK:   return [[CASTEXITARG0]] : tensor<!tf_type.variant<tensor<*xf32>>>

}

// -----
