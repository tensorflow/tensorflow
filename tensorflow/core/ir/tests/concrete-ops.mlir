// RUN: tfg-opt-no-passes %s | tfg-opt-no-passes | FileCheck %s

// CHECK: tfg.func @[[THEN_FUNC:.*]](
tfg.func @then_func(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>)
     -> (tensor<*xf32>) {
  return(%arg0) : tensor<*xf32>
}

// CHECK: tfg.func @[[ELSE_FUNC:.*]](
tfg.func @else_func(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>)
     -> (tensor<*xf32>) {
  return(%arg1) : tensor<*xf32>
}

// CHECK: tfg.func @[[COND_FUNC:.*]](
tfg.func @cond_func(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>)
     -> (tensor<*xi1>) {
  %True, %ctl = True : () -> (tensor<*xi1>)
  return(%True) : tensor<*xi1>
}

// CHECK: tfg.func @[[BODY_FUNC:.*]](
tfg.func @body_func(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>)
     -> (tensor<*xf32>, tensor<*xf32>) {
  %Ret:2, %ctl = Ret : () -> (tensor<*xf32>, tensor<*xf32>)
  return(%Ret#0, %Ret#1) : tensor<*xf32>, tensor<*xf32>
}

// CHECK: tfg.func @[[CASE_FUNC_0:.*]](
tfg.func @case_func_0(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>)
     -> (tensor<*xi32>, tensor<*xi64>) {
  %Ret:2, %ctl = Ret : () -> (tensor<*xi32>, tensor<*xi64>)
  return(%Ret#0, %Ret#1) : tensor<*xi32>, tensor<*xi64>
}

// CHECK: tfg.func @[[CASE_FUNC_1:.*]](
tfg.func @case_func_1(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>)
     -> (tensor<*xi32>, tensor<*xi64>) {
  %Ret:2, %ctl = Ret : () -> (tensor<*xi32>, tensor<*xi64>)
  return(%Ret#0, %Ret#1) : tensor<*xi32>, tensor<*xi64>
}

// CHECK: tfg.func @[[FOR_BODY:.*]](
tfg.func @for_body(%i: tensor<i32>, %arg0: tensor<*xf32>)
     -> (tensor<*xf32>) {
  return(%arg0) : tensor<*xf32>
}

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %[[ARGS:.*]]:3, %[[CTL0:.*]] = Op
  %args:2, %cond, %ctl_0 = Op : () -> (tensor<*xf32>, tensor<*xf32>, tensor<*xi1>)
  // CHECK: %[[IF:.*]], %[[CTL1:.*]] = If
  // CHECK-SAME: (%[[ARGS]]#2, %[[ARGS]]#0, %[[ARGS]]#1) [%[[CTL0]]] {
  %If, %ctl_1 = If(%cond, %args#0, %args#1) [%ctl_0] {
    // CHECK-SAME: Tcond = i1, Tin = [f32, f32], Tout = [f32],
    Tcond = i1, Tin = [f32, f32], Tout = [f32],
    // CHECK-SAME: else_branch = #tf_type.func<@[[ELSE_FUNC]], {}>
    else_branch = #tf_type.func<@else_func, {}>,
    // CHECK-SAME: output_shapes = [#tf_type.shape<>],
    output_shapes = [#tf_type.shape<>],
    // CHECK-SAME: then_branch = #tf_type.func<@[[THEN_FUNC]], {}>
    then_branch = #tf_type.func<@then_func, {}>
  }
  // CHECK-SAME: (tensor<*xi1>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  : (tensor<*xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  // CHECK: %[[WHILE:.*]]:2, %[[CTL2:.*]] = While(%[[ARGS]]#0, %[[IF]]) [%[[CTL0]], %[[CTL1]]] {
  %While:2, %ctl_2 = While(%args#0, %If) [%ctl_0, %ctl_1] {
    // CHECK-SAME: T = [f32, f32],
    T = [f32, f32],
    // CHECK-SAME: body = #tf_type.func<@[[BODY_FUNC]], {}>,
    body = #tf_type.func<@body_func, {}>,
    // CHECK-SAME: cond = #tf_type.func<@[[COND_FUNC]], {}>,
    cond = #tf_type.func<@cond_func, {}>,
    // CHECK-SAME: output_shapes = [#tf_type.shape<>, #tf_type.shape<>],
    output_shapes = [#tf_type.shape<>, #tf_type.shape<>],
    // CHECK-SAME: parallel_iterations = 10 : i64
    parallel_iterations = 10 : i64
  }
  // CHECK-SAME: (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)

  // CHECK: %[[INDEX:.*]]:3, %[[CT_3:.*]] = Index
  %Index:3, %ctl_3 = Index : () -> (tensor<i32>, tensor<i32>, tensor<i32>)

  // CHECK: %[[CASE:.*]]:2, %[[CTL4:.*]] = Case(%[[INDEX]]#0, %[[ARGS]]#0, %[[IF]]) [%[[CTL2]]] {
  // CHECK-SAME: Tin = [f32, f32], Tout = [i32, i64],
  // CHECK-SAME: branches = [#tf_type.func<@[[CASE_FUNC_0]], {}>,
  // CHECK-SAME:             #tf_type.func<@[[CASE_FUNC_1]], {}>],
  // CHECK-SAME: output_shapes = [#tf_type.shape<>, #tf_type.shape<>]}
  // CHECK-SAME: (tensor<i32>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xi32>, tensor<*xi64>)
  %Case:2, %ctl_4 = Case(%Index#0, %args#0, %If) [%ctl_2] {
    Tin = [f32, f32],
    Tout = [i32, i64],
    branches = [#tf_type.func<@case_func_0, {}>, #tf_type.func<@case_func_1, {}>],
    output_shapes = [#tf_type.shape<>, #tf_type.shape<>]
  }
  : (tensor<i32>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xi32>, tensor<*xi64>)

  // CHECK: %[[FOR:.*]], %[[CTL5:.*]] = For(%[[INDEX]]#0, %[[INDEX]]#1, %[[INDEX]]#2, %[[IF]])
  // CHECK-SAME: {T = [f32], body = #tf_type.func<@[[FOR_BODY]], {}>}
  // CHECK-SAME: (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
  %For, %ctl_5 = For(%Index#0, %Index#1, %Index#2, %If)
  {T = [f32], body = #tf_type.func<@for_body, {}>}
  : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)

  // Test that unknown condition functions are valid.
  // CHECK: %[[IF0:.*]], %[[CTL6:.*]] = If(%[[ARGS]]#2) {
  %If_0, %ctl_6 = If(%cond) {
    // CHECK-SAME: Tcond = i1, Tin = [], Tout = [f32],
    Tcond = i1, Tin = [], Tout = [f32],
    // CHECK-SAME: else_branch = #tf_type.func<@unknown_else_func, {}>
    else_branch = #tf_type.func<@unknown_else_func, {}>,
    // CHECK-SAME: output_shapes = [#tf_type.shape<>],
    output_shapes = [#tf_type.shape<>],
    // CHECK-SAME: then_branch = #tf_type.func<@unknown_then_func, {}>
    then_branch = #tf_type.func<@unknown_then_func, {}>
  }
  // CHECK-SAME: (tensor<*xi1>) -> (tensor<*xf32>)
  : (tensor<*xi1>) -> (tensor<*xf32>)

  // CHECK: %[[CASE0:.*]], %[[CTL7:.*]] = Case(%[[INDEX]]#0) {
  // CHECK-SAME: Tin = [], Tout = [i32],
  // CHECK-SAME: branches = [#tf_type.func<@unknown_case_func, {}>],
  // CHECK-SAME: output_shapes = [#tf_type.shape<>]}
  // CHECK-SAME: (tensor<i32>) -> (tensor<*xi32>)
  %Case_0, %ctl_7 = Case(%Index#0) {
    Tin = [],
    Tout = [i32],
    branches = [#tf_type.func<@unknown_case_func, {}>],
    output_shapes = [#tf_type.shape<>]
  } : (tensor<i32>) -> (tensor<*xi32>)

  // CHECK: %[[WHILE0:.*]], %[[CTL8:.*]] = While(%[[IF0]]) {
  %While_0, %ctl_8 = While(%If_0) {
    // CHECK-SAME: T = [f32],
    T = [f32],
    // CHECK-SAME: body = #tf_type.func<@unknown_body_func, {}>,
    body = #tf_type.func<@unknown_body_func, {}>,
    // CHECK-SAME: cond = #tf_type.func<@unknown_cond_func, {}>,
    cond = #tf_type.func<@unknown_cond_func, {}>,
    // CHECK-SAME: output_shapes = [#tf_type.shape<>],
    output_shapes = [#tf_type.shape<>],
    // CHECK-SAME: parallel_iterations = 10 : i64
    parallel_iterations = 10 : i64
  }
  // CHECK-SAME: (tensor<*xf32>) -> (tensor<*xf32>)
  : (tensor<*xf32>) -> (tensor<*xf32>)

  // CHECK: %[[FOR0:.*]], %[[CTL:.*]] = For(%[[INDEX]]#0, %[[INDEX]]#1, %[[INDEX]]#2, %[[IF]])
  // CHECK-SAME: {T = [f32], body = #tf_type.func<@unknown_for_body, {}>}
  // CHECK-SAME: (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
  %For_0, %ctl_9 = For(%Index#0, %Index#1, %Index#2, %If)
  {T = [f32], body = #tf_type.func<@unknown_for_body, {}>}
  : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)

  %Cast, %ctl_10 = Cast(%For_0) [%ctl_8] {SrcT = f32, DstT = f32} : (tensor<*xf32>) -> (tensor<f32>)
}
