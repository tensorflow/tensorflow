// RUN: tfg-transforms-opt -pass-pipeline='builtin.module(tfg.func(tfg-name-compress))' %s | FileCheck %s

// CHECK-LABEL: tfg.func @foo
// CHECK-SAME: tfg.name = "A"
// CHECK-NEXT: tfg.name = "B"
// CHECK-NEXT: -> 
// CHECK-SAME: tfg.name = "C"
// CHECK-NEXT: tfg.name = "D"
tfg.func @foo(%argument0: tensor<i1> {tfg.name = "argument0"},
              %argument1: tensor<i1> {tfg.name = "argument1"})
    -> (tensor<i1> {tfg.name = "result0"},
        tensor<i1> {tfg.name = "result1"}) {
  // CHECK: A({{.*}}) name("F")
  %A, %ctlA = A(%argument0) name("operation0") : (tensor<i1>) -> (tensor<i1>)
  // CHECK-NEXT: B({{.*}}) name("G")
  %B, %ctlB = B(%argument1) name("operation1") : (tensor<i1>) -> (tensor<i1>)
  // CHECK-NEXT: NoOp [{{.*}}] name("H")
  %ctlC = NoOp [%ctlA, %ctlB] name("operation2")
  // CHECK-NEXT: {tfg.name = "E"}
  return(%A, %B) [%ctlC {tfg.name = "control_result0"}] : tensor<i1>, tensor<i1>
}
