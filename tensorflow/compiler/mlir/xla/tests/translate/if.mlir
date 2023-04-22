// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK: [[R0:%.+]] ([[A0:.+]]: (f32[])) -> (f32[]) {
// CHECK:   %[[A0]] = (f32[]) parameter(0)
func @then_branch(%arg0: tuple<tensor<f32>>) -> tuple<tensor<f32>> {
  // CHECK:   %[[VAL0:.+]] = f32[] get-tuple-element((f32[]) %[[A0]]), index=0
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>>) -> tensor<f32>

  // CHECK:   %[[VAL1:.+]] = f32[] log(f32[] %[[VAL0]])
  %1 = "mhlo.log"(%0) : (tensor<f32>) -> tensor<f32>

  // CHECK:   ROOT %[[VAl2:.+]] = (f32[]) tuple(f32[] %[[VAL1]])
  %2 = "mhlo.tuple"(%1) : (tensor<f32>) -> tuple<tensor<f32>>
  return %2 : tuple<tensor<f32>>
}

// CHECK: [[R1:%.+]] ([[A0:.+]]: (f32[])) -> (f32[]) {
// CHECK:   %[[A0]] = (f32[]) parameter(0)
func @else_branch(%arg0: tuple<tensor<f32>>) -> tuple<tensor<f32>> {
  // CHECK:   %[[VAL0:.+]] = f32[] get-tuple-element((f32[]) %[[A0]]), index=0
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>>) -> tensor<f32>

  // CHECK:   %[[VAL1:.+]] = f32[] exponential(f32[] %[[VAL0]])
  %1 = "mhlo.exponential"(%0) : (tensor<f32>) -> tensor<f32>

  // CHECK:   ROOT %[[VAL2:.+]] = (f32[]) tuple(f32[] %[[VAL1]])
  %2 = "mhlo.tuple"(%1) : (tensor<f32>) -> tuple<tensor<f32>>
  return %2 : tuple<tensor<f32>>
}

// CHECK: ENTRY [[R3:%.+]] ([[A0:.+]]: f32[]) -> (f32[]) {
// CHECK:   %[[A0]] = f32[] parameter(0)
func @main(%arg0: tensor<f32>) -> tuple<tensor<f32>> {
  // CHECK:   %[[VAL0:.+]] = f32[] constant(10)
  %cst = constant  dense<1.000000e+01> : tensor<f32>

  // CHECK:   %[[VAL1:.+]] = pred[] compare(f32[] %[[A0]], f32[] %[[VAL0]]), direction=LT
  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  // CHECK:   %[[VAL2:.+]] = (f32[]) tuple(f32[] %[[A0]])
  %1 = "mhlo.tuple"(%arg0) : (tensor<f32>) -> tuple<tensor<f32>>

  // CHECK:   %[[VAL3:.+]] = (f32[]) conditional(pred[] %[[VAL1]], (f32[]) %[[VAL2]], (f32[]) %[[VAL2]]), true_computation=[[R0]], false_computation=[[R1]]
  %2 = "mhlo.if"(%0, %1, %1) ( {
  ^bb0(%arg1: tuple<tensor<f32>>):
    %6 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<f32>>) -> tensor<f32>
    %7 = "mhlo.log"(%6) : (tensor<f32>) -> tensor<f32>
    %8 = "mhlo.tuple"(%7) : (tensor<f32>) -> tuple<tensor<f32>>
    "mhlo.return"(%8) : (tuple<tensor<f32>>) -> ()
  },  {
  ^bb0(%arg1: tuple<tensor<f32>>):
    %6 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<f32>>) -> tensor<f32>
    %7 = "mhlo.exponential"(%6) : (tensor<f32>) -> tensor<f32>
    %8 = "mhlo.tuple"(%7) : (tensor<f32>) -> tuple<tensor<f32>>
    "mhlo.return"(%8) : (tuple<tensor<f32>>) -> ()
  }) : (tensor<i1>, tuple<tensor<f32>>, tuple<tensor<f32>>) -> tuple<tensor<f32>>

  // CHECK:   %[[VAL4:.+]] = f32[] get-tuple-element((f32[]) %[[VAL3]]), index=0
  %3 = "mhlo.get_tuple_element"(%2) {index = 0 : i32} : (tuple<tensor<f32>>) -> tensor<f32>

  // CHECK:   ROOT %[[VAL5:.+]] = (f32[]) tuple(f32[] %[[VAL4]])
  %4 = "mhlo.tuple"(%3) : (tensor<f32>) -> tuple<tensor<f32>>
  return %4 : tuple<tensor<f32>>
}
