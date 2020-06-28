// RUN: tf-opt %s -split-input-file -verify-diagnostics

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{unknown tf_saved_model dialect arg attribute 'tf_saved_model.not_a_real_arg_attr'}}
  func @f(%arg0: tensor<f32> {tf_saved_model.not_a_real_arg_attr = 1 : i32}) attributes {sym_visibility = "private"} {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{'tf_saved_model.bound_input' attribute should be a FlatSymbolRefAttr}}
  func @f(
    %arg0: tensor<f32> {tf_saved_model.bound_input = 1 : i32}
  ) attributes { tf_saved_model.exported_names = ["foo.some_func"] } {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{'tf_saved_model.bound_input' attribute must reference a valid symbol, got invalid symbol 'doesnt_exist'}}
  func @f(
    %arg0: tensor<f32> {tf_saved_model.bound_input = @doesnt_exist}
  ) attributes { tf_saved_model.exported_names = ["foo.some_func"] } {
    return
  }

}

// -----

// expected-error@+1 {{'tf_saved_model.exported_names' must be on an op whose immediate parent has attribute 'tf_saved_model.semantics'}}
func @f() attributes { tf_saved_model.exported_names = ["foo.some_func"] } {
  return
}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{'tf_saved_model.exported_names' must be on a 'func' or 'tf_saved_model.global_tensor' op}}
  "some_dialect.some_op"() {
    tf_saved_model.exported_names = ["foo"]
  } : () -> ()

}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{'tf_saved_model.exported_names' must be an array of strings}}
  func @f() attributes { tf_saved_model.exported_names = 1 : i32} {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-note@+1 {{previously seen here}}
  func @f() attributes { tf_saved_model.exported_names = ["foo"]} {
    return
  }

  // expected-error@+1 {{duplicate exported name 'foo'}}
  func @g() attributes { tf_saved_model.exported_names = ["foo"]} {
    return
  }

}

// -----

// expected-error@+1 {{'tf_saved_model.semantics' must be on a module op}}
"some_dialect.some_op"() {tf_saved_model.semantics} : () -> ()

// -----

// expected-error@+1 {{unknown tf_saved_model dialect attribute 'tf_saved_model.not_a_real_op_attr'}}
"some_dialect.some_op"() {tf_saved_model.not_a_real_op_attr} : () -> ()

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{'tf_saved_model.index_path' attribute should be an ArrayAttr}}
  func @f(
    %arg0: tensor<f32> {tf_saved_model.index_path = 1}
  ) attributes { tf_saved_model.exported_names = ["f"] } {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{'tf_saved_model.index_path' elements should be strings or 64-bit integers}}
  func @f(
    %arg0: tensor<f32> {tf_saved_model.index_path = [1.0] }
  ) attributes { tf_saved_model.exported_names = ["f"] } {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{all arguments should have 'tf_saved_model.index_path', 'tf_saved_model.bound_input' or 'tf.resource_name' attributes}}
  func @f(
    %arg0: tensor<f32>
  ) attributes { tf_saved_model.exported_names = ["f"] } {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  "tf_saved_model.global_tensor"() { sym_name = "some_constant", type = tensor<f32>, value = dense<42.0> : tensor<f32> } : () -> ()

  // expected-error@+1 {{all 'tf_saved_model.index_path' arg attributes should precede all 'tf_saved_model.bound_input' arg attributes}}
  func @f(
    %arg0: tensor<f32> {tf_saved_model.bound_input = @some_constant},
    %arg1: tensor<f32> {tf_saved_model.index_path = [0]}
  ) attributes { tf_saved_model.exported_names = ["f"] } {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{all results should have 'tf_saved_model.index_path' attributes}}
  func @f() -> tensor<f32>
  attributes { tf_saved_model.exported_names = ["f"] } {
    %ret = "some_dialect.some_op"() : () -> tensor<f32>
    return %ret : tensor<f32>
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // Sanity-check that we are verifying tf_saved_model.index_path attributes
  // on results as well. The underlying verification logic is shared,
  // so no need to test all error cases.

  // expected-error@+1 {{'tf_saved_model.index_path' elements should be strings or 64-bit integers}}
  func @f() -> (tensor<f32> {tf_saved_model.index_path = [1.0]})
  attributes { tf_saved_model.exported_names = ["f"] } {
    %ret = "some_dialect.some_op"() : () -> tensor<f32>
    return %ret : tensor<f32>
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  func @f() attributes { tf_saved_model.exported_names = ["f"] } {
    // expected-error@+1 {{exported function cannot be internally referenced}}
    "some_dialect.some_call"() { callee = @g } : () -> ()
    return
  }

  // expected-note@+1 {{references this exported function}}
  func @g() attributes { tf_saved_model.exported_names = ["g"] } {
    return
  }

}

// -----

// expected-error@+1 {{modules with 'tf_saved_model.semantics' must have analyzable symbol uses}}
module attributes {tf_saved_model.semantics} {

  func @root() attributes {tf_saved_model.exported_names = ["root"]} {
    "some_unregistered_dialect.maybe_a_symbol_table"() ({
      return
    }) : () -> ()
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {
  // expected-error@+1 {{'type' and 'value' attributes should have compatible tensor types}}
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v0", type = tensor<3xf32>, value = dense<42.0> : tensor<9xf32> } : () -> ()
}

// -----

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.0> : tensor<f32> } : () -> ()
  // expected-error@+1 {{duplicate 'tf_saved_model.bound_input' binding}}
  func @f(
    %arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v},
    %arg1: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}
  ) attributes {tf_saved_model.exported_names = ["f"]} {
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  // expected-error@+1 {{can only apply 'tf_saved_model' argument attributes to exported functions}}
  func @f(%arg0: tensor<!tf.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v})
  -> (tensor<?xf32> {tf_saved_model.index_path = []}) attributes {sym_visibility = "private"} {
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<?xf32>>>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  // expected-error@+1 {{bound input with type 'tensor<f32>' expected to have type 'tensor<!tf.resource<tensor<?xf32>>>'}}
  func @f(%arg0: tensor<f32> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{'type' attribute for immutable 'tf_saved_model.global_tensor' should have a static shape}}
  "tf_saved_model.global_tensor"() { sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{the initializer function does not exist}}
  "tf_saved_model.session_initializer"() { initializer = @init } : () -> ()
}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{the initializer function should have no output}}
  "tf_saved_model.session_initializer"() { initializer = @init } : () -> ()
  func @init() -> tensor<1xf32> attributes {sym_visibility = "private"} {
    %0 = "tf.Const"() {value = dense<[1.0]> : tensor<1xf32> } : () -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  "tf_saved_model.session_initializer"() { initializer = @init } : () -> ()
  // expected-error@+1 {{there must be no more than one session_initializer op}}
  "tf_saved_model.session_initializer"() { initializer = @init } : () -> ()
  func @init() -> tensor<1xf32> attributes {sym_visibility = "private"} {
    %0 = "tf.Const"() {value = dense<[1.0]> : tensor<1xf32> } : () -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{exported function @f should be public}}
  func @f(
    %arg0: tensor<f32> {tf.resource_name = "resource"}
  ) attributes { sym_visibility = "private", tf_saved_model.exported_names = ["foo.some_func"] } {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{non-exported function @f should be private}}
  func @f(
    %arg0: tensor<f32> {tf.resource_name = "resource"}
  ) {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{the initializer function does not exist}}
  "tf_saved_model.session_initializer"() { initializer = @init } : () -> ()
}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{the initializer function should have no output}}
  "tf_saved_model.session_initializer"() { initializer = @init } : () -> ()
  func @init() -> (tensor<1xf32> {tf_saved_model.index_path = ["output"]})
    attributes { tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"] } {
    %0 = "tf.Const"() {value = dense<[1.0]> : tensor<1xf32> } : () -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  "tf_saved_model.session_initializer"() { initializer = @init } : () -> ()
  // expected-error@+1 {{there must be no more than one session_initializer op}}
  "tf_saved_model.session_initializer"() { initializer = @init } : () -> ()
  func @init() -> (tensor<1xf32> {tf_saved_model.index_path = ["output"]})
    attributes { tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"] } {
    %0 = "tf.Const"() {value = dense<[1.0]> : tensor<1xf32> } : () -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{the initializer function should be exported}}
  "tf_saved_model.session_initializer"() { initializer = @init } : () -> ()
  func @init() attributes {sym_visibility = "private"} {
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{the initializer function should have only one exported name}}
  "tf_saved_model.session_initializer"() { initializer = @init } : () -> ()
  func @init() attributes { tf_saved_model.exported_names = ["a", "b"] } {
    return
  }
}
