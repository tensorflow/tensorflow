// RUN: tfg-transforms-opt -pass-pipeline='builtin.module(tfg.func(tfg-cse))' %s | FileCheck %s

// CHECK-LABEL: tfg.func @test_simple_cse
// CHECK-SAME: %[[A:.*]]: tensor
tfg.func @test_simple_cse(%a: tensor<i32> {tfg.name = "a0"}) 
    -> (tensor<i32> {tfg.name = "b0"},
        tensor<i32> {tfg.name = "b1"}) {
  // CHECK: %[[ADD1:.*]], %{{.*}} = Add(%[[A]], %[[A]]) name("add1")
  %Add1, %ctl1 = Add(%a, %a) name("add1") : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  // CHECK-NOT: Add(%[[A]], %[[A]]) name("add0")
  %Add0, %ctl0 = Add(%a, %a) name("add0") : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  // CHECK: return(%[[ADD1]], %[[ADD1]])
  return(%Add0, %Add1) : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: tfg.func @test_cse_across_regions
// CHECK-SAME: %[[A:.*]]: tensor
// CHECK-NEXT: %[[COND:.*]]: tensor
tfg.func @test_cse_across_regions(%a: tensor<i32> {tfg.name = "a0"},
                                  %cond: tensor<i1> {tfg.name = "cond"})
    -> (tensor<i32> {tfg.name = "b0"},
        tensor<i32> {tfg.name = "b1"}) {
  // CHECK: %[[ADD0:.*]], %{{.*}} = Add(%[[A]], %[[A]]) name("add0")
  %Add0, %ctl0 = Add(%a, %a) name("add0") : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  // CHECK: %[[IF:.*]], %{{.*}} = StatelessIfRegion
  %If, %ctl = StatelessIfRegion %cond then {
    // CHECK-NOT: Add(%[[A]], %[[A]]) name("add1")
    %Add1, %ctl1 = Add(%a, %a) name("add1") : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
    // CHECK: yield(%[[ADD0]])
    yield(%Add1) : tensor<i32>
  } else {
    // CHECK-NOT: Add(%[[A]], %[[A]]) name("add2")
    %Add2, %ctl2 = Add(%a, %a) name("add1") : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
    // CHECK: yield(%[[ADD0]])
    yield(%Add2) : tensor<i32>
  } {_mlir_name = "if"} : (tensor<i1>) -> (tensor<i32>)
  // CHECK: return(%[[IF]], %[[ADD0]])
  return(%If, %Add0) : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: tfg.func @test_cse_control_tokens
// CHECK-SAME: %[[A:.*]]: tensor
tfg.func @test_cse_control_tokens(%a: tensor<i32> {tfg.name = "a0"}) 
    -> (tensor<i32> {tfg.name = "b0"}) {
  // CHECK: %[[ADD1:.*]], %[[CTL1:.*]] = Add(%[[A]], %[[A]]) name("add1")
  %Add1, %ctl1 = Add(%a, %a) name("add1") : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  // CHECK-NOT: Add(%[[A]], %[[A]]) name("add0")
  %Add0, %ctl0 = Add(%a, %a) name("add0") : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  // CHECK: %[[CTL:.*]] = NoOp [%[[CTL1]], %[[CTL1]]]
  %ctl = NoOp [%ctl1, %ctl0]
  // CHECK: return(%[[ADD1]]) [%[[CTL]] {
  return(%Add1) [%ctl {tfg.name = "noop"}] : tensor<i32>
}
