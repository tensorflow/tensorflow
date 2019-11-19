// RUN: tf-opt -xla-legalize-control-flow %s -o - | FileCheck %s

// CHECK-LABEL: func @while(%arg0: tensor<i64>) -> tensor<i64> {
func @while(%arg0: tensor<i64>) -> tensor<i64> {
  //CHECK:   br ^bb1(%arg0 : tensor<i64>)
  //CHECK: ^bb1([[VAL0:%.+]]: tensor<i64>):
  //CHECK:   [[VAL1:%.+]] = "xla_hlo.compare"([[VAL0]], [[VAL0]])
  //CHECK:   [[VAL2:%.+]] = extract_element [[VAL1]][] : tensor<i1>
  //CHECK:   cond_br [[VAL2]], ^bb2([[VAL0]] : tensor<i64>), ^bb3([[VAL0]] : tensor<i64>)
  //CHECK: ^bb2([[VAL3:%.+]]: tensor<i64>):
  //CHECK:   [[VAL4:%.+]] = xla_hlo.add [[VAL3]], [[VAL3]]
  //CHECK:   br ^bb1([[VAL4]] : tensor<i64>)
  //CHECK: ^bb3([[VAL5:%.+]]: tensor<i64>):
  %0 = "xla_hlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i64>):
    %1 = "xla_hlo.compare"(%arg1, %arg1) {comparison_direction = "LT", name = "compare.2"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "xla_hlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %1 = xla_hlo.add %arg1, %arg1 {name = "compare.0"} : tensor<i64>
    "xla_hlo.return"(%1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>

  // CHECK-NEXT:   return [[VAL5]]
  return %0 : tensor<i64>
}

// CHECK-LABEL: func @conditional
func @conditional(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK:   [[C0:%.+]] = constant dense<1.000000e+01> : tensor<f32>
  %cst = constant dense<1.000000e+01> : tensor<f32>

  // CHECK:   [[VAL0:%.+]] = "xla_hlo.compare"(%arg0, [[C0]]) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %0 = "xla_hlo.compare"(%arg0, %cst) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  // CHECK:   [[VAL1:%.+]] = extract_element [[VAL0]][] : tensor<i1>
  // CHECK:   cond_br [[VAL1]], ^bb1(%arg0 : tensor<f32>), ^bb2(%arg0 : tensor<f32>)
  %1 = "xla_hlo.conditional"(%0, %arg0, %arg0) ( {

  ^bb0(%arg1: tensor<f32>):
    // CHECK: ^bb1([[VAL2:%.+]]: tensor<f32>):
    // CHECK:   [[VAL3:%.+]] = "xla_hlo.log"([[VAL2]]) : (tensor<f32>) -> tensor<f32>
    // CHECK:   br ^bb3([[VAL3]] : tensor<f32>)
    %2 = "xla_hlo.log"(%arg1) : (tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  },  {

  ^bb0(%arg1: tensor<f32>):
    // CHECK: ^bb2([[VAL4:%.+]]: tensor<f32>):
    // CHECK:   [[VAL5:%.+]] = "xla_hlo.exp"([[VAL4]]) : (tensor<f32>) -> tensor<f32>
    // CHECK:   br ^bb3([[VAL5]] : tensor<f32>)
    %2 = "xla_hlo.exp"(%arg1) : (tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: ^bb3([[VAL6:%.+]]: tensor<f32>):
  // CHECK:   return [[VAL6]] : tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: func @while_with_multiple_blocks_in_body(%arg0: tensor<i64>) -> tensor<i64> {
func @while_with_multiple_blocks_in_body(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK:   br ^[[COND_ENTRY:.+]](%arg0 : tensor<i64>)
  // CHECK: ^[[COND_ENTRY]](%0: tensor<i64>):
  // CHECK:   %1 = "xla_hlo.compare"(%0, %0) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK:   %2 = extract_element %1[] : tensor<i1>
  // CHECK:   cond_br %2, ^[[BODY_ENTRY:.+]](%0 : tensor<i64>), ^[[EXIT:.+]](%0 : tensor<i64>)
  // CHECK: ^[[BODY_ENTRY]](%3: tensor<i64>):
  // CHECK:   br ^[[BODY_SUCC:.+]](%3 : tensor<i64>)
  // CHECK: ^[[BODY_SUCC]](%4: tensor<i64>):
  // CHECK:   %5 = xla_hlo.add %4, %4 : tensor<i64>
  // CHECK:   br ^[[COND_ENTRY]](%5 : tensor<i64>)
  // CHECK: ^[[EXIT]](%6: tensor<i64>):
  // CHECK:   return %6 : tensor<i64>
  // CHECK: }
  %0 = "xla_hlo.while"(%arg0) ( {
  ^cond_entry(%arg1: tensor<i64>):
    %1 = "xla_hlo.compare"(%arg1, %arg1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "xla_hlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^body_entry(%arg1: tensor<i64>):
    br ^body_succ(%arg1: tensor<i64>)
  ^body_succ(%0: tensor<i64>):
    %1 = xla_hlo.add %0, %0 : tensor<i64>
    "xla_hlo.return"(%1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>

  return %0 : tensor<i64>
}

// CHECK-LABEL: func @while_with_multiple_blocks_in_cond(%arg0: tensor<i64>) -> tensor<i64> {
func @while_with_multiple_blocks_in_cond(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK:   br ^[[COND_ENTRY:.+]](%arg0 : tensor<i64>)
  // CHECK: ^[[COND_ENTRY]](%0: tensor<i64>):
  // CHECK:   br ^[[COND_SUCC:.+]](%0 : tensor<i64>)
  // CHECK: ^[[COND_SUCC]](%1: tensor<i64>):
  // CHECK:   %2 = "xla_hlo.compare"(%1, %1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK:   %3 = extract_element %2[] : tensor<i1>
  // CHECK:   cond_br %3, ^[[BODY_ENTRY:.+]](%0 : tensor<i64>), ^[[EXIT:.+]](%0 : tensor<i64>)
  // CHECK: ^[[BODY_ENTRY]](%4: tensor<i64>):
  // CHECK:   br ^[[COND_ENTRY]](%4 : tensor<i64>)
  // CHECK: ^[[EXIT]](%5: tensor<i64>):
  // CHECK:   return %5 : tensor<i64>
  // CHECK: }
  %0 = "xla_hlo.while"(%arg0) ( {
  ^cond_entry(%arg1: tensor<i64>):
    br ^cond_succ(%arg1: tensor<i64>)
  ^cond_succ(%0: tensor<i64>):
    %1 = "xla_hlo.compare"(%0, %0) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "xla_hlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^body_entry(%arg1: tensor<i64>):
    "xla_hlo.return"(%arg1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>

  return %0 : tensor<i64>
}

// CHECK-LABEL: func @conditional_with_multiple_blocks(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> tensor<f32> {
func @conditional_with_multiple_blocks(%arg0: tensor<f32>, %arg1: tensor<f32>, %pred: tensor<i1>) -> tensor<f32> {
  // CHECK:   %0 = extract_element %arg2[] : tensor<i1>
  // CHECK:   cond_br %0, ^[[THEN_ENTRY:.+]](%arg0 : tensor<f32>), ^[[ELSE_ENTRY:.+]](%arg1 : tensor<f32>)
  // CHECK: ^[[THEN_ENTRY]](%1: tensor<f32>):
  // CHECK:   br ^[[THEN_SUCC:.+]](%1 : tensor<f32>)
  // CHECK: ^[[THEN_SUCC]](%2: tensor<f32>):
  // CHECK:   %3 = "xla_hlo.log"(%2) : (tensor<f32>) -> tensor<f32>
  // CHECK:   br ^[[EXIT:.+]](%3 : tensor<f32>)
  // CHECK: ^[[ELSE_ENTRY]](%4: tensor<f32>):
  // CHECK:   %5 = "xla_hlo.exp"(%4) : (tensor<f32>) -> tensor<f32>
  // CHECK:   br ^[[EXIT]](%5 : tensor<f32>)
  // CHECK: ^[[EXIT]](%6: tensor<f32>):
  // CHECK:   return %6 : tensor<f32>
  // CHECK: }
  %1 = "xla_hlo.conditional"(%pred, %arg0, %arg1) ( {
  ^then_entry(%arg2: tensor<f32>):
    br ^then_succ(%arg2: tensor<f32>)
  ^then_succ(%0: tensor<f32>):
    %2 = "xla_hlo.log"(%0) : (tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  },  {
  ^else_entry(%arg2: tensor<f32>):
    %2 = "xla_hlo.exp"(%arg2) : (tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}
