// RUN: mlir-opt -test-legalize-patterns -test-legalize-mode=full -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @multi_level_mapping
func @multi_level_mapping() {
  // CHECK: "test.type_producer"() : () -> f64
  // CHECK: "test.type_consumer"(%{{.*}}) : (f64) -> ()
  %result = "test.type_producer"() : () -> i32
  "test.type_consumer"(%result) : (i32) -> ()
  "test.return"() : () -> ()
}

// Test that operations that are erased don't need to be legalized.
// CHECK-LABEL: func @dropped_region_with_illegal_ops
func @dropped_region_with_illegal_ops() {
  // CHECK-NEXT: test.return
  "test.drop_region_op"() ({
    %ignored = "test.illegal_op_f"() : () -> (i32)
    "test.return"() : () -> ()
  }) : () -> ()
  "test.return"() : () -> ()
}
// CHECK-LABEL: func @replace_non_root_illegal_op
func @replace_non_root_illegal_op() {
  // CHECK-NEXT: "test.legal_op_b"
  // CHECK-NEXT: test.return
  %result = "test.replace_non_root"() : () -> (i32)
  "test.return"() : () -> ()
}

// -----

// Test that children of recursively legal operations are ignored.
func @recursively_legal_invalid_op() {
  /// Operation that is statically legal.
  module attributes {test.recursively_legal} {
    %ignored = "test.illegal_op_f"() : () -> (i32)
  }
  /// Operation that is dynamically legal, i.e. the function has a pattern
  /// applied to legalize the argument type before it becomes recursively legal.
  func @dynamic_func(%arg: i64) attributes {test.recursively_legal} {
    %ignored = "test.illegal_op_f"() : () -> (i32)
    "test.return"() : () -> ()
  }

  "test.return"() : () -> ()
}

// -----

// Test that region cloning can be properly undone.
func @test_undo_region_clone() {
  "test.region"() ({
    ^bb1(%i0: i64):
      "test.invalid"(%i0) : (i64) -> ()
  }) {legalizer.should_clone} : () -> ()

  // expected-error@+1 {{failed to legalize operation 'test.illegal_op_f'}}
  %ignored = "test.illegal_op_f"() : () -> (i32)
  "test.return"() : () -> ()
}
