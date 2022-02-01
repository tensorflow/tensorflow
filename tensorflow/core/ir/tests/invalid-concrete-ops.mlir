// RUN: tfg-opt-no-passes %s --split-input-file --verify-diagnostics

// expected-note@+1 {{see referenced function}}
tfg.func @body(%arg0: tensor<i32>) -> (tensor<*xf32>) {
  %Ret, %ctl = Ret : () -> (tensor<*xf32>)
  return(%Ret) : tensor<*xf32>
}

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Index, %Arg, %ctl = Op : () -> (tensor<i32>, tensor<*xf32>)
  // expected-error@+1 {{body function expected to have 4 arguments but got 2}}
  %For, %ctl_0 = For(%Index, %Index, %Index, %Arg)
  {T = [f32], body = #tf_type.func<@body, {}>}
  : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
}

// -----

// expected-note@+1 {{see referenced function}}
tfg.func @body(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> (tensor<*xf32>) {
  return(%arg1) : tensor<*xf32>
}

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Index, %Arg, %ctl = Op : () -> (tensor<i32>, tensor<*xf32>)
  // expected-error@+1 {{body function expected to have 2 return values but got 1}}
  %For:2, %ctl_0 = For(%Index, %Index, %Index, %Arg)
  {T = [f32], body = #tf_type.func<@body, {}>}
  : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
}

// -----

// expected-note@+1 {{see referenced function}}
tfg.func @body(%arg0: tensor<f32>, %arg1: tensor<*xf32>) -> (tensor<*xf32>) {
  return(%arg1) : tensor<*xf32>
}

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Index, %Arg, %ctl = Op : () -> (tensor<i32>, tensor<*xf32>)
  // expected-error@+1 {{body function argument #0 dtype 'f32' does not match}}
  %For, %ctl_0 = For(%Index, %Index, %Index, %Arg)
  {T = [f32], body = #tf_type.func<@body, {}>}
  : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
}

// -----

// expected-note@+1 {{see referenced function}}
tfg.func @body(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> (tensor<i32>) {
  return(%arg0) : tensor<i32>
}

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Index, %Arg, %ctl = Op : () -> (tensor<i32>, tensor<*xf32>)
  // expected-error@+1 {{body function result #0 dtype 'i32' does not match}}
  %For, %ctl_0 = For(%Index, %Index, %Index, %Arg)
  {T = [f32], body = #tf_type.func<@body, {}>}
  : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
}

// -----

// expected-note@+1 {{see referenced function}}
tfg.func @else() -> (tensor<f32>) {
  %Ret, %ctl = Ret : () -> (tensor<f32>)
  return(%Ret) : tensor<f32>
}

tfg.func @test(%arg0: tensor<i1>) -> (tensor<i32>) {
  // expected-error@+1 {{else function result #0 dtype 'f32' does not match}}
  %If, %ctl = If(%arg0) {
    Tcond = i1, Tin = [], Tout = [i32],
    else_branch = #tf_type.func<@else, {}>,
    then_branch = #tf_type.func<@then, {}>,
    output_shapes = [#tf_type.shape<>]
  } : (tensor<i1>) -> (tensor<i32>)
  return(%If) : tensor<i32>
}

// -----

// expected-note@+1 {{see referenced function}}
tfg.func @then() -> (tensor<f32>) {
  %Ret, %ctl = Ret : () -> (tensor<f32>)
  return(%Ret) : tensor<f32>
}

tfg.func @test(%arg0: tensor<i1>) -> (tensor<i32>) {
  // expected-error@+1 {{then function result #0 dtype 'f32' does not match}}
  %If, %ctl = If(%arg0) {
    Tcond = i1, Tin = [], Tout = [i32],
    else_branch = #tf_type.func<@else, {}>,
    then_branch = #tf_type.func<@then, {}>,
    output_shapes = [#tf_type.shape<>]
  } : (tensor<i1>) -> (tensor<i32>)
  return(%If) : tensor<i32>
}

// -----

// expected-note@+1 {{see referenced function}}
tfg.func @case() -> (tensor<f32>) {
  %Ret, %ctl = Ret : () -> (tensor<f32>)
  return(%Ret) : tensor<f32>
}

tfg.func @test(%arg0: tensor<i32>) -> (tensor<i32>) {
  // expected-error@+1 {{branch #0 function result #0 dtype 'f32' does not match}}
  %Case, %ctl = Case(%arg0) {
    Tin = [],
    Tout = [i32],
    branches = [#tf_type.func<@case, {}>],
    output_shapes = [#tf_type.shape<>]
  } : (tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}

// -----

tfg.func @cond(%arg0: tensor<i32>) -> (tensor<i1>) {
  %Ret, %ctl = Ret : () -> (tensor<i1>)
  return(%Ret) : tensor<i1>
}

// expected-note@+1 {{see referenced function}}
tfg.func @body(%arg0: tensor<f32>) -> (tensor<f32>) {
  return(%arg0) : tensor<f32>
}

tfg.func @test(%arg0: tensor<i32>) -> (tensor<i32>) {
  // expected-error@+1 {{body function argument #0 dtype 'f32' does not match}}
  %While, %ctl = While(%arg0) {
    T = [i32], body = #tf_type.func<@body, {}>, cond = #tf_type.func<@cond, {}>,
    output_shapes = [#tf_type.shape<>], parallel_iterations = 10 : i64
  } : (tensor<i32>) -> (tensor<i32>)
  return(%While) : tensor<i32>
}

// -----

tfg.func @body(%arg0: tensor<i32>) -> (tensor<i1>) {
  %Ret, %ctl = Ret : () -> (tensor<i1>)
  return(%Ret) : tensor<i1>
}

// expected-note@+1 {{see referenced function}}
tfg.func @cond(%arg0: tensor<f32>) -> (tensor<f32>) {
  return(%arg0) : tensor<f32>
}

tfg.func @test(%arg0: tensor<i32>) -> (tensor<i32>) {
  // expected-error@+1 {{cond function argument #0 dtype 'f32' does not match}}
  %While, %ctl = While(%arg0) {
    T = [i32], body = #tf_type.func<@body, {}>, cond = #tf_type.func<@cond, {}>,
    output_shapes = [#tf_type.shape<>], parallel_iterations = 10 : i64
  } : (tensor<i32>) -> (tensor<i32>)
  return(%While) : tensor<i32>
}
