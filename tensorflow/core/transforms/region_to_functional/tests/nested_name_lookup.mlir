// RUN: tfg-transforms-opt --split-input-file --tfg-region-to-functional %s | FileCheck %s

// Check that argument name lookup when converting a region op nested inside
// a loop region op; check that the lookup is able to "see through" block
// arguments.

// CHECK-LABEL: @test
// CHECK-SAME: %[[START:.*]]: tensor
// CHECK-NEXT: %[[END:.*]]: tensor
// CHECK-NEXT: %[[STEP:.*]]: tensor
// CHECK-NEXT: %[[A:.*]]: tensor
tfg.func @test_for(%start: tensor<i32> {tfg.name = "start"},
               %end: tensor<i32> {tfg.name = "end"},
               %step: tensor<i32> {tfg.name = "step"},
               %a: tensor<i32> {tfg.name = "a"}) -> (tensor<i32>) {
  // CHECK: For(%[[START]], %[[END]], %[[STEP]], %[[A]])
  %For, %ctlFor = ForRegion(%a) from %start to %end by %step {
  ^bb0(%idx: tensor<i32>, %arg: tensor<i32>, %ctlIdx: !tf_type.control, %ctlArg: !tf_type.control):
    %Cond, %ctlCond = Cond name("cond") : () -> (tensor<i1>)
    %If, %ctlIf = IfRegion %Cond then {
      yield(%arg) : tensor<i32>
    } else {
      yield(%idx) : tensor<i32>
    } {_mlir_name = "if"} : (tensor<i1>) -> (tensor<i32>)
    yield(%If) : tensor<i32>
  } {_mlir_name = "for"} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%For) : tensor<i32>
}

// CHECK-LABEL: @if_then_function
// CHECK-SAME: tfg.name = "a"
// CHECK-NEXT: tfg.name = "start"

// CHECK-LABEL: @if_else_function
// CHECK-SAME: tfg.name = "a"
// CHECK-NEXT: tfg.name = "start"

// CHECK-LABEL: @for_body_function
// CHECK-SAME: tfg.name = "start"
// CHECK-NEXT: tfg.name = "a"

// -----

// CHECK-LABEL: @test_while
// CHECK-SAME: %[[A:.*]]: tensor
tfg.func @test_while(%a: tensor<i32> {tfg.name = "a"}) -> (tensor<i32>) {
  // CHECK: While(%[[A]])
  %While, %ctlWhile = WhileRegion(%a) {
  ^b(%arg: tensor<i32>, %ctlArg: !tf_type.control):
    %Cond, %ctlCond = Cond name("cond") : () -> (tensor<i1>)
    condition %Cond : tensor<i1> (%arg) : tensor<i32>
  } do {
  ^b(%arg: tensor<i32>, %ctlArg: !tf_type.control):
    %Cond, %ctlCond = Cond name("cond_body") : () -> (tensor<i1>)
    %If, %ctlIf = IfRegion %Cond then {
      yield(%arg) : tensor<i32>
    } else {
      yield(%arg) : tensor<i32>
    } : (tensor<i1>) -> (tensor<i32>)
    yield(%If) : tensor<i32>
  } {parallel_iterations = 10 : i64, _mlir_name = "while"} : (tensor<i32>) -> (tensor<i32>)
  return(%While) : tensor<i32>
}

// CHECK-LABEL: @if_then_function
// CHECK-SAME: tfg.name = "a"

// CHECK-LABEL: @if_else_function
// CHECK-SAME: tfg.name = "a"

// CHECK-LABEL: @while_body_function
// CHECK-SAME: tfg.name = "a"
