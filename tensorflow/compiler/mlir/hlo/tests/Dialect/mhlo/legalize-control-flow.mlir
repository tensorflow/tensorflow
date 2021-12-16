// RUN: mlir-hlo-opt -mhlo-legalize-control-flow %s -o - | FileCheck %s

// CHECK-LABEL: func @while(%arg0: tensor<i64>) -> tensor<i64> {
func @while(%arg0: tensor<i64>) -> tensor<i64> {
  //CHECK:   br ^bb1(%arg0 : tensor<i64>)
  //CHECK: ^bb1([[VAL0:%.+]]: tensor<i64>):
  //CHECK:   [[VAL1:%.+]] = "mhlo.compare"([[VAL0]], [[VAL0]])
  //CHECK:   [[VAL2:%.+]] = tensor.extract [[VAL1]][] : tensor<i1>
  //CHECK:   cond_br [[VAL2]], ^bb2([[VAL0]] : tensor<i64>), ^bb3([[VAL0]] : tensor<i64>)
  //CHECK: ^bb2([[VAL3:%.+]]: tensor<i64>):
  //CHECK:   [[VAL4:%.+]] = mhlo.add [[VAL3]], [[VAL3]]
  //CHECK:   br ^bb1([[VAL4]] : tensor<i64>)
  //CHECK: ^bb3([[VAL5:%.+]]: tensor<i64>):
  %0 = "mhlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = "LT", name = "compare.2"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %1 = mhlo.add %arg1, %arg1 {name = "compare.0"} : tensor<i64>
    "mhlo.return"(%1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>

  // CHECK-NEXT:   return [[VAL5]]
  return %0 : tensor<i64>
}

// CHECK-LABEL: func @conditional
func @conditional(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK:   [[C0:%.+]] = arith.constant dense<1.000000e+01> : tensor<f32>
  %cst = arith.constant dense<1.000000e+01> : tensor<f32>

  // CHECK:   [[VAL0:%.+]] = "mhlo.compare"(%arg0, [[C0]]) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  // CHECK:   [[VAL1:%.+]] = tensor.extract [[VAL0]][] : tensor<i1>
  // CHECK:   cond_br [[VAL1]], ^bb1(%arg0 : tensor<f32>), ^bb2(%arg0 : tensor<f32>)
  %1 = "mhlo.if"(%0, %arg0, %arg0) ( {

  ^bb0(%arg1: tensor<f32>):
    // CHECK: ^bb1([[VAL2:%.+]]: tensor<f32>):
    // CHECK:   [[VAL3:%.+]] = "mhlo.log"([[VAL2]]) : (tensor<f32>) -> tensor<f32>
    // CHECK:   br ^bb3([[VAL3]] : tensor<f32>)
    %2 = "mhlo.log"(%arg1) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  },  {

  ^bb0(%arg1: tensor<f32>):
    // CHECK: ^bb2([[VAL4:%.+]]: tensor<f32>):
    // CHECK:   [[VAL5:%.+]] = "mhlo.exponential"([[VAL4]]) : (tensor<f32>) -> tensor<f32>
    // CHECK:   br ^bb3([[VAL5]] : tensor<f32>)
    %2 = "mhlo.exponential"(%arg1) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: ^bb3([[VAL6:%.+]]: tensor<f32>):
  // CHECK:   return [[VAL6]] : tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: func @case2
// Test the two branches case as the common. Following tests verify degenerate
// behavior.
func @case2(%arg0 : tensor<i32>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK:   %[[TARGET:.*]] = tensor.extract %arg0[] : tensor<i32>
  // CHECK:   %[[CMP_BR_0:.*]] = arith.constant 0 : i32
  // CHECK:   %[[PRED:.*]] = arith.cmpi eq, %[[TARGET]], %[[CMP_BR_0]]
  // CHECK:   cond_br %[[PRED]], ^bb1, ^bb2
  // CHECK: ^bb1:
  // CHECK:   %[[BR0_RESULT:.*]] = "mhlo.log"(%arg1)
  // CHECK:   br ^bb3(%[[BR0_RESULT]] : tensor<4xf32>)
  // CHECK: ^bb2:
  // CHECK:   %[[BR1_RESULT:.*]] = "mhlo.exponential"(%arg2)
  // CHECK:   br ^bb3(%[[BR1_RESULT]] : tensor<4xf32>)
  // CHECK: ^bb3(%[[RESULT:.*]]: tensor<4xf32>):
  // CHECK-NOT: mhlo.case
  // CHECK:   return %[[RESULT]]
  %1 = "mhlo.case"(%arg0, %arg1, %arg2) ( {
    ^bb0(%phi0 : tensor<4xf32>):
      %2 = "mhlo.log"(%phi0) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%2) : (tensor<4xf32>) -> ()
  }, {
    ^bb0(%phi1 : tensor<4xf32>):
      %3 = "mhlo.exponential"(%phi1) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%3) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @case3
func @case3(%arg0 : tensor<i32>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>, %arg3 : tensor<4xf32>) -> tensor<4xf32> {
  // Just testing that the condition blocks are setup correctly.
  // Blocks:
  //   bb1: Condition block 1
  //   bb2: Branch body 0
  //   bb3: Branch body 1
  //   bb4: Branch body 2
  //   bb5: Tail
  // CHECK:   %[[PRED0:.*]] = arith.cmpi eq
  // CHECK:   cond_br %[[PRED0]], ^bb2, ^bb1
  // CHECK: ^bb1:
  // CHECK:   %[[C1:.*]] = arith.constant 1 : i32
  // CHECK:   %[[PRED1:.*]] = arith.cmpi eq, %{{.*}}, %c1_i32 : i32
  // CHECK:   cond_br %[[PRED1]], ^bb3, ^bb4
  %1 = "mhlo.case"(%arg0, %arg1, %arg2, %arg3) ( {
    ^bb0(%phi0 : tensor<4xf32>):
      %2 = "mhlo.log"(%phi0) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%2) : (tensor<4xf32>) -> ()
  }, {
    ^bb0(%phi1 : tensor<4xf32>):
      %3 = "mhlo.exponential"(%phi1) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%3) : (tensor<4xf32>) -> ()
  }, {
    ^bb0(%phi2 : tensor<4xf32>):
      %3 = "mhlo.floor"(%phi2) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%3) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @case0
func @case0(%arg0 : tensor<i32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK:   br ^bb1
  // CHECK: ^bb1:
  // CHECK:   %[[BR0_RESULT:.*]] = "mhlo.log"
  // CHECK:   br ^bb2(%[[BR0_RESULT]] : tensor<4xf32>)
  // CHECK: ^bb2(%[[RESULT:.*]]: tensor<4xf32>):
  // CHECK:   return %[[RESULT]]
  %1 = "mhlo.case"(%arg0, %arg1) ( {
    ^bb0(%phi0 : tensor<4xf32>):
      %2 = "mhlo.log"(%phi0) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%2) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}
