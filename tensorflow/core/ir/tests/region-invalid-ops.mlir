// RUN: tfg-opt-no-passes --split-input-file --verify-diagnostics %s

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %cond, %fwd, %ctl = Op : () -> (tensor<*xi32>, tensor<*xf32>)
  // expected-error@+1 {{operand #0 must be tensor of 1-bit signless integer values}}
  %IfRegion, %ctl_0 = IfRegion %cond then {
    yield(%fwd) [%ctl] : tensor<*xf32>
  } else {
    yield(%fwd) : tensor<*xf32>
  } : (tensor<*xi32>) -> (tensor<*xf32>)
}

// -----

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %cond, %fwd, %ctl = Op : () -> (tensor<*xi1>, tensor<*xf32>)
  // expected-error@+1 {{then region must be terminated by a 'tfg.yield'}}
  %IfRegion, %ctl_0 = IfRegion %cond then {
    return(%fwd) [%ctl] : tensor<*xf32>
  } else {
    yield(%fwd) : tensor<*xf32>
  } : (tensor<*xi1>) -> (tensor<*xf32>)
}

// -----

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %cond, %fwd, %ctl = Op : () -> (tensor<*xi1>, tensor<*xf32>)
  // expected-error@+1 {{else region must be terminated by a 'tfg.yield'}}
  %IfRegion, %ctl_0 = IfRegion %cond then {
    yield(%fwd) [%ctl] : tensor<*xf32>
  } else {
    return(%fwd) : tensor<*xf32>
  } : (tensor<*xi1>) -> (tensor<*xf32>)
}

// -----

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Index, %ctl = Index : () -> (tensor<i32>)
  %A, %ctl_0 = A : () -> (tensor<f32>)
  // expected-error@+1 {{branch region #0 is not terminated by a 'tfg.yield'}}
  %CaseRegion, %ctl_1 = CaseRegion %Index {
    return(%A) : tensor<f32>
  } : (tensor<i32>) -> (tensor<f32>)
}

// -----

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Index, %ctl = Index : () -> (tensor<i32>)
  %A, %ctl_0 = A : () -> (tensor<f32>)
  // expected-error@+1 {{has 1 regions but 2 branch function attributes}}
  %CaseRegion, %ctl_1 = CaseRegion %Index {
    yield(%A) : tensor<f32>
  } {branch_attrs = [{}, {}]} : (tensor<i32>) -> (tensor<f32>)
}

// -----

// Test body yield must have same operand types.
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Op, %ctl = Op : () -> (tensor<*xi32>)
  // expected-error@+1 {{condition region must be terminated by a 'tfg.condition'}}
  %WhileRegion, %ctl_0 = WhileRegion(%Op) [%ctl] {
  ^bb0(%arg0: tensor<*xi32>, %arg1: !tf_type.control):
    %cond, %ctl_1 = Op : () -> (tensor<*xi1>)
    yield(%arg0) : tensor<*xi32>
  } do {
  ^bb0(%arg0: tensor<*xi32>, %arg1: !tf_type.control):
    yield(%arg0) [%arg1] : tensor<*xi32>
  } {parallel_iterations = 10 :i64} : (tensor<*xi32>) -> (tensor<*xi32>)
}

// -----

// Test body yield must have same operand types.
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Op, %ctl = Op : () -> (tensor<*xi32>)
  // expected-error@+1 {{body region must be terminated by a 'tfg.yield'}}
  %WhileRegion, %ctl_0 = WhileRegion(%Op) [%ctl] {
  ^bb0(%arg0: tensor<*xi32>, %arg1: !tf_type.control):
    %cond, %ctl_1 = Op : () -> (tensor<*xi1>)
    condition %cond : tensor<*xi1> (%arg0) : tensor<*xi32>
  } do {
  ^bb0(%arg0: tensor<*xi32>, %arg1: !tf_type.control):
    %Cond, %ctl_2 = Cond : () -> (tensor<*xi1>)
    condition %Cond : tensor<*xi1> (%arg0) : tensor<*xi32>
  } {parallel_iterations = 10 :i64} : (tensor<*xi32>) -> (tensor<*xi32>)
}

// -----

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Index, %ctl = Index : () -> (tensor<i32>)
  %Arg, %ctl_0 = Arg : () -> (tensor<*xf32>)
  // expected-error@+1 {{expected first body block argument to be an i32 tensor}}
  %For, %ctl_1 = ForRegion(%Arg) [%ctl] from %Index to %Index by %Index {
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<*xf32>, %arg2: !tf_type.control, %arg3: !tf_type.control):
    yield(%arg1) [%arg3] : tensor<*xf32>
  } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
}

// -----

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Index, %ctl = Index : () -> (tensor<i32>)
  %Arg, %ctl_0 = Arg : () -> (tensor<*xf32>)
  // expected-error@+1 {{body region must be terminated by a 'tfg.yield'}}
  %For, %ctl_1 = ForRegion(%Arg) [%ctl] from %Index to %Index by %Index {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<*xf32>, %arg2: !tf_type.control, %arg3: !tf_type.control):
    return(%arg1) [%arg3] : tensor<*xf32>
  } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
}

// -----

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Index, %ctl = Index : () -> (tensor<i32>)
  %Arg, %ctl_0 = Arg : () -> (tensor<*xf32>)
  // expected-error@+1 {{expected the body block to have at least have the loop index as an argument}}
  %ctl_1 = ForRegion [%ctl] from %Index to %Index by %Index {
  ^bb0:
    yield
  } : (tensor<i32>, tensor<i32>, tensor<i32>)
}

// -----

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Index, %ctl = Index : () -> (tensor<i32>)
  %Arg, %ctl_0 = Arg : () -> (tensor<*xf32>)
  // expected-error@+1 {{expected same number of data values and control tokens}}
  %For, %ctl_1 = ForRegion(%Arg) [%ctl] from %Index to %Index by %Index {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<*xf32>, %arg2: !tf_type.control):
    yield(%arg1) [%arg2] : tensor<*xf32>
  } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
}

// -----

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Index, %ctl = Index : () -> (tensor<i32>)
  // expected-error@+1 {{should not be a control token}}
  %For, %ctl_1 = ForRegion(%Index) [%ctl] from %Index to %Index by %Index {
  ^bb0(%arg0: tensor<i32>, %arg1: !tf_type.control, %arg2: !tf_type.control, %arg3: tensor<*xf32>):
    yield(%arg0) [%arg2] : tensor<i32>
  } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)
}
