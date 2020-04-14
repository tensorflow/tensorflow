// RUN: tf-opt -tf-saved-model-mark-func-visibility -symbol-dce -split-input-file %s | FileCheck %s --dump-input=fail

module attributes {tf_saved_model.semantics} {

  // Test case: Unused function should be deleted.

  // CHECK-NOT: func @unused
  func @unused() {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Root calls child. Child should not be deleted.

  // CHECK: func @root
  func @root() attributes {tf_saved_model.exported_names = ["root"]} {
    "tf.some_call"() { callee = @child } : () -> ()
    return
  }

  // CHECK: func @child
  func @child() {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Don't crash if attribute that doesn't reference a func.

  "tf.some_opaque_global_variable"() { sym_name = "some_global" } : () -> ()

  func @root2() attributes {tf_saved_model.exported_names = ["root2"]} {
    "tf.do_something_with_a_global"() { global = @some_global } : () -> ()
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Delete recursively dead cycle.

  // CHECK-NOT: func @recursively_dead0
  func @recursively_dead0() {
    "tf.some_call"() { callee = @recursively_dead1 } : () -> ()
    return
  }
  // CHECK-NOT: func @recursively_dead1
  func @recursively_dead1() {
    "tf.some_call"() { callee = @recursively_dead0 } : () -> ()
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Root calls child with a deeply nested symbol reference.
  // Child should not be deleted.

  // CHECK: func @root
  func @root() attributes {tf_saved_model.exported_names = ["root"]} {
    "tf.some_call"() {callee = {callee = {callee = @child}}} : () -> ()
    return
  }

  // CHECK: func @child
  func @child() {
    return
  }

}

// -----

// Test case: If the module doesn't have tf_saved_model semantics, then this
// pass shouldn't do anything.
module {
  // CHECK: func @not_dead()
  func @not_dead() {
    return
  }
}
