// RUN: xla-opt -xla-legalize-tf-control-flow %s | FileCheck %s

// CHECK-LABEL: @if
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>, [[ARG1:%.+]]: tensor<f32>)
func @if(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>) {
  // CHECK: [[VAL0:%.+]] = "mhlo.compare"([[ARG0]], [[ARG1]]) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: [[VAL1:%.+]] = "mhlo.tuple"([[ARG0]], [[ARG1]])
  // CHECK: [[VAL2:%.+]] = "mhlo.if"([[VAL0]], [[VAL1]], [[VAL1]]) ( {
  // CHECK: ^bb0([[THEN_ARG:%.+]]: tuple<tensor<f32>, tensor<f32>>):
  // CHECK:   [[VAL4:%.+]] = "mhlo.get_tuple_element"([[THEN_ARG]]) {index = 0 : i32}
  // CHECK:   [[VAL5:%.+]] = "mhlo.get_tuple_element"([[THEN_ARG]]) {index = 1 : i32}
  // CHECK:   [[VAL6:%.+]] = call @cond_true([[VAL4]], [[VAL5]])
  // CHECK:   [[VAL7:%.+]] = "mhlo.tuple"([[VAL6]])
  // CHECK:   "mhlo.return"([[VAL7]]) : (tuple<tensor<f32>>) -> ()
  // CHECK: },  {
  // CHECK: ^bb0([[ELSE_ARG:%.+]]: tuple<tensor<f32>, tensor<f32>>)
  // CHECK:   [[VAL4:%.+]] = "mhlo.get_tuple_element"([[ELSE_ARG]]) {index = 0 : i32}
  // CHECK:   [[VAL5:%.+]] = "mhlo.get_tuple_element"([[ELSE_ARG]]) {index = 1 : i32}
  // CHECK:   [[VAL6:%.+]] = call @cond_false([[VAL4]], [[VAL5]])
  // CHECK:   [[VAL7:%.+]] = "mhlo.tuple"([[VAL6]])
  // CHECK:   "mhlo.return"([[VAL7]]) : (tuple<tensor<f32>>) -> ()
  // CHECK: })
  %1 = "tf.If"(%0, %arg0, %arg1) {else_branch = @cond_false, is_stateless = true, then_branch = @cond_true} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: [[VAL3:%.+]] = "mhlo.get_tuple_element"([[VAL2]]) {index = 0 : i32}
  // CHECK: return [[VAL3]]
  return %1 : tensor<f32>
}

func @cond_false(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>
attributes  {tf._input_shapes = ["tfshape$", "tfshape$"]} {
  %0 = "mhlo.exponential"(%arg1) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @cond_true(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>
attributes  {tf._input_shapes = ["tfshape$", "tfshape$"]} {
  %0 = "mhlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}


// CHECK-LABEL: @ifRegion
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>, [[ARG1:%.+]]: tensor<f32>)
func @ifRegion(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>) {
  // CHECK: [[VAL0:%.+]] = "mhlo.compare"([[ARG0]], [[ARG1]]) {comparison_direction = "GT"}
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: [[VAL1:%.+]] = "mhlo.tuple"([[ARG0]])
  // CHECK: [[VAL2:%.+]] = "mhlo.tuple"([[ARG1]])
  // CHECK: [[VAL3:%.+]] = "mhlo.if"([[VAL0]], [[VAL1]], [[VAL2]]) ( {
  %1 = "tf.IfRegion"(%0) ( {
  // CHECK: ^{{[a-z0-9]+}}([[TRUE_ARG:%.+]]: tuple<tensor<f32>>):
    // CHECK: [[VAL5:%.+]] = "mhlo.get_tuple_element"([[TRUE_ARG]]) {index = 0 : i32}
    // CHECK: [[VAL6:%.+]] = "mhlo.log"([[VAL5]])
    %2 = "mhlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
    // CHECK: [[VAL7:%.+]] = "mhlo.tuple"([[VAL6]])
    // CHECK: "mhlo.return"([[VAL7]])
    "tf.Yield"(%2) : (tensor<f32>) -> ()
  }, {
  // CHECK: ^{{[a-z0-9]+}}([[FALSE_ARG:%.+]]: tuple<tensor<f32>>):
    // CHECK: [[VAL5:%.+]] = "mhlo.get_tuple_element"([[FALSE_ARG]]) {index = 0 : i32}
    // CHECK: [[VAL6:%.+]] = "mhlo.exponential"([[VAL5]])
    %2 = "mhlo.exponential"(%arg1) : (tensor<f32>) -> tensor<f32>
    // CHECK: [[VAL7:%.+]] = "mhlo.tuple"([[VAL6]])
    // CHECK: "mhlo.return"([[VAL7]])
    "tf.Yield"(%2) : (tensor<f32>) -> ()
  // CHECK: }) : (tensor<i1>, tuple<tensor<f32>>, tuple<tensor<f32>>) -> tuple<tensor<f32>>
  }) {is_stateless = true} : (tensor<i1>) -> tensor<f32>
  // CHECK: [[VAL4:%.+]] = "mhlo.get_tuple_element"([[VAL3]]) {index = 0 : i32}
  // CHECK: return [[VAL4]]
  return %1 : tensor<f32>
}


// CHECK-LABEL: func @case
// CHECK-SAME:  %[[BRANCH_INDEX:.*]]: tensor<i32>, %[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>) -> (tensor<f32>, tensor<f32>)
func @case(%index: tensor<i32>, %arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = "tf.Case"(%index, %arg0, %arg1) {branches = [@exponential, @log, @floor], is_stateless = true} : (tensor<i32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  // CHECK: %[[TUPLE_INPUT:.*]] = "mhlo.tuple"(%[[ARG0]], %[[ARG1]]) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>>
  // CHECK: %[[CASE:.*]]:2 = "mhlo.case"(%[[BRANCH_INDEX]], %[[TUPLE_INPUT]], %[[TUPLE_INPUT]], %[[TUPLE_INPUT]]) ( {
  // CHECK:   ^bb0(%[[TUPLE_ARG:.*]]: tuple<tensor<f32>, tensor<f32>>):
  // CHECK:     %[[TUPLE_ELEMENT_0:.*]] = "mhlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[TUPLE_ELEMENT_1:.*]] = "mhlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 1 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[CALL_EXP:.*]]:2 = call @exponential(%[[TUPLE_ELEMENT_0]], %[[TUPLE_ELEMENT_1]]) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  // CHECK:     "mhlo.return"(%[[CALL_EXP]]#0, %[[CALL_EXP]]#1) : (tensor<f32>, tensor<f32>) -> ()
  // CHECK:   },  {
  // CHECK:   ^bb0(%[[TUPLE_ARG:.*]]: tuple<tensor<f32>, tensor<f32>>):
  // CHECK:     %[[TUPLE_ELEMENT_0:.*]] = "mhlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[TUPLE_ELEMENT_1:.*]] = "mhlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 1 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[CALL_LOG:.*]]:2 = call @log(%[[TUPLE_ELEMENT_0]], %[[TUPLE_ELEMENT_1]]) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  // CHECK:     "mhlo.return"(%[[CALL_LOG]]#0, %[[CALL_LOG]]#1) : (tensor<f32>, tensor<f32>) -> ()
  // CHECK:   },  {
  // CHECK:   ^bb0(%[[TUPLE_ARG:.*]]: tuple<tensor<f32>, tensor<f32>>):
  // CHECK:     %[[TUPLE_ELEMENT_0:.*]] = "mhlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[TUPLE_ELEMENT_1:.*]] = "mhlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 1 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[CALL_FLOOR:.*]]:2 = call @floor(%[[TUPLE_ELEMENT_0]], %[[TUPLE_ELEMENT_1]]) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  // CHECK:     "mhlo.return"(%[[CALL_FLOOR]]#0, %[[CALL_FLOOR]]#1) : (tensor<f32>, tensor<f32>) -> ()
  // CHECK:   }) : (tensor<i32>, tuple<tensor<f32>, tensor<f32>>, tuple<tensor<f32>, tensor<f32>>, tuple<tensor<f32>, tensor<f32>>) -> (tensor<f32>, tensor<f32>)
  return %0#0, %0#1 : tensor<f32>, tensor<f32>
// CHECK:   return %[[CASE]]#0, %[[CASE]]#1 : tensor<f32>, tensor<f32>
}

func @exponential(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0 = "mhlo.exponential"(%arg1) : (tensor<f32>) -> tensor<f32>
  return %0, %arg1 : tensor<f32>, tensor<f32>
}

func @log(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0 = "mhlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0, %arg1 : tensor<f32>, tensor<f32>
}

func @floor(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0 = "mhlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0, %arg1 : tensor<f32>, tensor<f32>
}


// CHECK-LABEL: func @caseRegion
// CHECK-SAME:  ([[BRANCH_INDEX:%.+]]: tensor<i32>, [[ARG0:.+]]: tensor<f32>, [[ARG1:%.+]]: tensor<f32>)
func @caseRegion(%index: tensor<i32>, %arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  // CHECK: [[VAL0:%.+]] = "mhlo.tuple"([[ARG1]])
  // CHECK: [[VAL1:%.+]] = "mhlo.tuple"([[ARG0]], [[ARG1]])
  // CHECK: [[VAL2:%.+]] = "mhlo.tuple"([[ARG0]], [[ARG1]])
  // CHECK: [[VAL3:%.+]]:2 = "mhlo.case"([[BRANCH_INDEX]], [[VAL0]], [[VAL1]], [[VAL2]]) ( {
  %0:2 = "tf.CaseRegion"(%index) ( {
  // CHECK: ^{{[a-z0-9]+}}([[BRANCH0_ARG:%.+]]: tuple<tensor<f32>>):
    // CHECK: [[VAL4:%.+]] = "mhlo.get_tuple_element"([[BRANCH0_ARG]]) {index = 0 : i32}
    // CHECK: [[VAL5:%.+]] = "mhlo.exponential"([[VAL4]])
    %1 = "mhlo.exponential"(%arg1) : (tensor<f32>) -> tensor<f32>
    // CHECK: "mhlo.return"([[VAL5]], [[VAL4]])
    "tf.Yield"(%1, %arg1) : (tensor<f32>, tensor<f32>) -> ()
  }, {
  // CHECK: ^{{[a-z0-9]+}}([[BRANCH1_ARG:%.+]]: tuple<tensor<f32>, tensor<f32>>):
    // CHECK: [[VAL4:%.+]] = "mhlo.get_tuple_element"([[BRANCH1_ARG]]) {index = 0 : i32}
    // CHECK: [[VAL5:%.+]] = "mhlo.get_tuple_element"([[BRANCH1_ARG]]) {index = 1 : i32}
    // CHECK: [[VAL6:%.+]] = "mhlo.log"([[VAL4]])
    %1 = "mhlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
    // CHECK: "mhlo.return"([[VAL6]], [[VAL5]])
    "tf.Yield"(%1, %arg1) : (tensor<f32>, tensor<f32>) -> ()
  }, {
  // CHECK: ^{{[a-z0-9]+}}([[BRANCH2_ARG:%.+]]: tuple<tensor<f32>, tensor<f32>>):
    // CHECK: [[VAL4:%.+]] = "mhlo.get_tuple_element"([[BRANCH2_ARG]]) {index = 0 : i32}
    // CHECK: [[VAL5:%.+]] = "mhlo.get_tuple_element"([[BRANCH2_ARG]]) {index = 1 : i32}
    // CHECK: [[VAL6:%.+]] = "mhlo.floor"([[VAL4]])
    %1 = "mhlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
    // CHECK: "mhlo.return"([[VAL6]], [[VAL5]])
    "tf.Yield"(%1, %arg1) : (tensor<f32>, tensor<f32>) -> ()
  // CHECK: }) : (tensor<i32>, tuple<tensor<f32>>, tuple<tensor<f32>, tensor<f32>>, tuple<tensor<f32>, tensor<f32>>) -> (tensor<f32>, tensor<f32>)
  }) {is_stateless = true} : (tensor<i32>) -> (tensor<f32>, tensor<f32>)
  // CHECK: return [[VAL3]]#0, [[VAL3]]#1 : tensor<f32>, tensor<f32>
  return %0#0, %0#1 : tensor<f32>, tensor<f32>
}


// CHECK-LABEL: func @while
func @while() -> tensor<i32> {
  // CHECK-DAG: [[VAL0:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[VAL1:%.+]] = mhlo.constant dense<-1>
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<-1> : tensor<i32>
  // CHECK: [[VAL2:%.+]]:3 = "mhlo.while"([[VAL0]], [[VAL1]], [[VAL0]]) ( {
  // CHECK: ^bb0([[COND_ARG0:%.+]]: tensor<i32>, [[COND_ARG1:%.+]]: tensor<i32>, [[COND_ARG2:%.+]]: tensor<i32>):
  // CHECK:   [[VAL3:%.+]] = call @while_cond([[COND_ARG0]], [[COND_ARG1]], [[COND_ARG2]])
  // CHECK:   "mhlo.return"([[VAL3]])
  // CHECK: },  {
  // CHECK: ^bb0([[BODY_ARG0:%.+]]: tensor<i32>, [[BODY_ARG1:%.+]]: tensor<i32>, [[BODY_ARG2:%.+]]: tensor<i32>):
  // CHECK:   [[VAL3:%.+]]:3 = call @while_body([[BODY_ARG0]], [[BODY_ARG1]], [[BODY_ARG2]])
  // CHECK:   "mhlo.return"([[VAL3]]#0, [[VAL3]]#1, [[VAL3]]#2)
  // CHECK: }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  // CHECK: return [[VAL2]]#2
  %2:3 = "tf.While"(%0, %1, %0) {body = @while_body, cond = @while_cond, is_stateless = true, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  return %2#2 : tensor<i32>
}
func @while_cond(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.constant dense<10> : tensor<i32>
  %1 = "mhlo.compare"(%arg2, %0) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %1 : tensor<i1>
}
func @while_body(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.add %arg2, %0 : tensor<i32>
  %2 = mhlo.add %arg0, %0 : tensor<i32>
  return %2, %arg1, %1 : tensor<i32>, tensor<i32>, tensor<i32>
}


// CHECK-LABEL: func @whileRegion
func @whileRegion() -> tensor<i32> {
  // CHECK: [[VAL0:%.+]] = mhlo.constant dense<0>
  %0 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: [[VAL1:%.+]] = mhlo.constant dense<-1>
  %1 = mhlo.constant dense<-1> : tensor<i32>
  // CHECK: [[VAL2:%.+]]:3 = "mhlo.while"([[VAL0]], [[VAL1]], [[VAL0]]) ( {
  %2:3 = "tf.WhileRegion"(%0, %1, %0) ( {
  // CHECK: ^bb0([[COND_ARG0:%.+]]: tensor<i32>, [[COND_ARG1:%.+]]: tensor<i32>, [[COND_ARG2:%.+]]: tensor<i32>):
  ^cond(%carg0: tensor<i32>, %carg1: tensor<i32>, %carg2: tensor<i32>):
    // CHECK: [[VAL3:%.+]] = mhlo.constant dense<10>
    %3 = mhlo.constant dense<10> : tensor<i32>
    // CHECK: [[VAL4:%.+]] = "mhlo.compare"([[COND_ARG2]], [[VAL3]]) {comparison_direction = "LT"}
    %4 = "mhlo.compare"(%carg2, %3) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    // CHECK: "mhlo.return"([[VAL4]])
    "tf.Yield"(%4) : (tensor<i1>) -> ()
  }, {
  // CHECK: ^bb0([[BODY_ARG0:%.+]]: tensor<i32>, [[BODY_ARG1:%.+]]: tensor<i32>, [[BODY_ARG2:%.+]]: tensor<i32>):
  ^body(%barg0: tensor<i32>, %barg1: tensor<i32>, %barg2: tensor<i32>):
    // CHECK: [[VAL5:%.+]] = mhlo.constant dense<1>
    %5 = mhlo.constant dense<1> : tensor<i32>
    // CHECK: [[VAL6:%.+]] = mhlo.add [[BODY_ARG2]], [[VAL5]]
    %6 = mhlo.add %barg2, %5 : tensor<i32>
    // CHECK: [[VAL7:%.+]] = mhlo.add [[BODY_ARG0]], [[VAL5]]
    %7 = mhlo.add %barg0, %5 : tensor<i32>
    // CHECK: "mhlo.return"([[VAL7]], [[BODY_ARG1]], [[VAL6]])
    "tf.Yield"(%7, %barg1, %6) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  // CHECK: }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  // CHECK: return [[VAL2]]#2
  return %2#2 : tensor<i32>
}


// CHECK-LABEL: func @whileRegionImplicitInputs
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @whileRegionImplicitInputs(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: [[VAL0:%.+]] = mhlo.constant dense<0>
  %0 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: [[VAL1:%.+]] = mhlo.constant dense<-1>
  %1 = mhlo.constant dense<-1> : tensor<i32>
  // CHECK: [[VAL2:%.+]]:3 = "mhlo.while"([[ARG0]], [[VAL0]], [[VAL1]]) ( {
  %2 = "tf.WhileRegion"(%arg0) ( {
  // CHECK: ^bb0([[COND_ARG1:%.+]]: tensor<i32>, [[COND_ARG2:%.+]]: tensor<i32>, [[COND_ARG3:%.+]]: tensor<i32>):
  ^cond(%carg0: tensor<i32>):
    // CHECK: [[VAL3:%.+]] = "mhlo.compare"([[COND_ARG1]], [[COND_ARG2]]) {comparison_direction = "LT"}
    %3 = "mhlo.compare"(%carg0, %0) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    // CHECK: "mhlo.return"([[VAL3]])
    "tf.Yield"(%3) : (tensor<i1>) -> ()
  }, {
  // CHECK: ^bb0([[BODY_ARG1:%.+]]: tensor<i32>, [[BODY_ARG2:%.+]]: tensor<i32>, [[BODY_ARG3:%.+]]: tensor<i32>):
  ^body(%barg0: tensor<i32>):
    // CHECK: [[VAL3:%.+]] = mhlo.add [[BODY_ARG1]], [[BODY_ARG3]]
    %3 = mhlo.add %barg0, %1 : tensor<i32>
    // CHECK: [[VAL4:%.+]] = mhlo.add [[BODY_ARG1]], [[VAL3]]
    %4 = mhlo.add %barg0, %3 : tensor<i32>
    // CHECK: "mhlo.return"([[VAL4]], [[BODY_ARG2]], [[BODY_ARG3]])
    "tf.Yield"(%4) : (tensor<i32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>
  // CHECK: }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  // CHECK: return [[VAL2]]#0
  return %2 : tensor<i32>
}


// CHECK-LABEL: func @whileRegionMultipleImplicitInputs
func @whileRegionMultipleImplicitInputs() {
  // CHECK: [[VAL0:%.+]] = mhlo.constant dense<0>
  %0 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: [[VAL1:%.+]] = mhlo.constant dense<-1>
  %1 = mhlo.constant dense<-1> : tensor<i32>
  // CHECK: [[VAL2:%.+]]:2 = "mhlo.while"([[VAL0]], [[VAL1]]) ( {
  "tf.WhileRegion"() ( {
  // CHECK: ^bb0([[COND_ARG0:%.+]]: tensor<i32>, [[COND_ARG1:%.+]]: tensor<i32>):
    // CHECK: [[VAL3:%.+]] = "mhlo.compare"([[COND_ARG0]], [[COND_ARG1]]) {comparison_direction = "LT"}
    %2 = "mhlo.compare"(%0, %1) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    // CHECK: "mhlo.return"([[VAL3]])
    "tf.Yield"(%2) : (tensor<i1>) -> ()
  }, {
  // CHECK: ^bb0([[BODY_ARG0:%.+]]: tensor<i32>, [[BODY_ARG1:%.+]]: tensor<i32>):
    // CHECK: [[VAL3:%.+]] = mhlo.add [[BODY_ARG0]], [[BODY_ARG1]]
    %2 = mhlo.add %0, %1 : tensor<i32>
    // CHECK: "mhlo.return"([[BODY_ARG0]], [[BODY_ARG1]])
    "tf.Yield"() : () -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : () -> ()
  // CHECK: }) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  // CHECK: return
  return
}
