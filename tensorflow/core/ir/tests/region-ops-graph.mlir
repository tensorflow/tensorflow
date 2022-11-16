// RUN: tfg-opt-no-passes %s | tfg-opt-no-passes | FileCheck %s

// Test that all of the regions control-flow ops allow graph regions.

// CHECK-LABEL: tfg.func @test_if
tfg.func @test_if(%cond: tensor<i1>, %arg: tensor<i32>) -> (tensor<i32>) {
  // CHECK: IfRegion
  %If, %ctlIf = IfRegion %cond then {
    // CHECK: %[[A:.*]], %{{.*}} = A(%[[B:.*]]) :
    %A, %ctlA = A(%B) : (tensor<i32>) -> (tensor<i32>)
    // CHECK: %[[B]], %{{.*}} = B
    %B, %ctlB = B(%arg) : (tensor<i32>) -> (tensor<i32>)
    yield(%A) : tensor<i32>
  } else {
    // CHECK: %[[A:.*]], %{{.*}} = A(%[[B:.*]]) :
    %A, %ctlA = A(%B) : (tensor<i32>) -> (tensor<i32>)
    // CHECK: %[[B]], %{{.*}} = B
    %B, %ctlB = B(%arg) : (tensor<i32>) -> (tensor<i32>)
    yield(%A) : tensor<i32>
  } : (tensor<i1>) -> (tensor<i32>)
  return(%If) : tensor<i32>
}

// CHECK-LABEL: tfg.func @test_case
tfg.func @test_case(%idx: tensor<i32>, %arg: tensor<i32>) -> (tensor<i32>) {
  // CHECK: CaseRegion
  %Case, %ctlCase = CaseRegion %idx {
    // CHECK: %[[A:.*]], %{{.*}} = A(%[[B:.*]]) :
    %A, %ctlA = A(%B) : (tensor<i32>) -> (tensor<i32>)
    // CHECK: %[[B]], %{{.*}} = B
    %B, %ctlB = B(%arg) : (tensor<i32>) -> (tensor<i32>)
    yield(%A) : tensor<i32>
  } : (tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}

// CHECK-LABEL: tfg.func @test_while
tfg.func @test_while(%arg: tensor<i32>) -> (tensor<i32>) {
  // CHECK: WhileRegion
  %While, %ctl = WhileRegion(%arg) {
  ^bb0(%arg0: tensor<i32>, %arg1: !tf_type.control):
    // CHECK: %[[TRUE:.*]], %{{.*}} = True(%[[A:.*]]) :
    %True, %ctlTrue = True(%A) : (tensor<i32>) -> (tensor<i1>)
    // CHECK: %[[A]], %{{.*}} = A
    %A, %ctlA = A(%arg0) : (tensor<i32>) -> (tensor<i32>)
    condition %True : tensor<i1> (%arg0) : tensor<i32>
  } do {
  ^bb0(%arg0: tensor<i32>, %arg1: !tf_type.control):
    // CHECK: %[[A:.*]], %{{.*}} = A(%[[B:.*]]) :
    %A, %ctlA = A(%B) : (tensor<i32>) -> (tensor<i32>)
    // CHECK: %[[B]], %{{.*}} = B
    %B, %ctlB = B(%arg) : (tensor<i32>) -> (tensor<i32>)
    yield(%A) : tensor<i32>
  } {parallel_iterations = 10 : i64} : (tensor<i32>) -> (tensor<i32>)
  return(%While) : tensor<i32>
}

// CHECK-LABEL: tfg.func @test_for
tfg.func @test_for(%start: tensor<i32>, %limit: tensor<i32>, %delta: tensor<i32>, %arg: tensor<i32>) -> (tensor<i32>) {
  // CHECK: ForRegion
  %For, %ctl = ForRegion(%arg) from %start to %limit by %delta {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: !tf_type.control, %arg3: !tf_type.control):
    // CHECK: %[[A:.*]], %{{.*}} = A(%[[B:.*]]) :
    %A, %ctlA = A(%B) : (tensor<i32>) -> (tensor<i32>)
    // CHECK: %[[B]], %{{.*}} = B
    %B, %ctlB = B(%arg) : (tensor<i32>) -> (tensor<i32>)
    yield(%A) : tensor<i32>
  } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%For) : tensor<i32>
}

