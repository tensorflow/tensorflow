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

