// RUN: tf-opt %s -split-input-file -verify-diagnostics

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{unknown tf_saved_model dialect arg attribute 'tf_saved_model.not_a_real_arg_attr'}}
  func @f(
    %arg0: tensor<f32> {tf_saved_model.not_a_real_arg_attr = 1 : i32}
  ) attributes { tf_saved_model.exported_names = ["foo.some_func"] } {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // expected-error@+1 {{'tf_saved_model.bound_input' attribute should be a SymbolRefAttr}}
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

