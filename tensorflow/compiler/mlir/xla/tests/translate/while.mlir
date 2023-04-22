// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s -o - | FileCheck %s

module {
  func @main(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = "mhlo.while"(%arg0) ( {
    // CHECK: [[R0:%.+]] ([[A0:.+]]: s64[]) -> s64[] {
    // CHECK:   %[[A0]] = s64[] parameter(0)
    // CHECK:   ROOT %add.4 = s64[] add(s64[] %[[A0]], s64[] %[[A0]])
    // CHECK: [[R1:%.+]] ([[A0:.+]]: s64[]) -> pred[] {
    // CHECK:   %[[A0]] = s64[] parameter(0)
    // CHECK:   ROOT %compare.7 = pred[] compare(s64[] %[[A0]], s64[] %[[A0]]), direction=LT
    ^bb0(%arg1: tensor<i64>):
      %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "mhlo.return"(%1) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg1: tensor<i64>):
      %1 = mhlo.add %arg1, %arg1 : tensor<i64>
      "mhlo.return"(%1) : (tensor<i64>) -> ()
    }) : (tensor<i64>) -> tensor<i64>

    // CHECK: ENTRY %main.9 ([[A0:.+]]: s64[]) -> s64[] {
    // CHECK:   %[[A0]] = s64[] parameter(0)
    // CHECK:   ROOT %while.8 = s64[] while(s64[] %[[A0]]), condition=[[R1]], body=[[R0]]
    return %0 : tensor<i64>
  }
}
