// RUN: mlir-opt -split-input-file -test-legalize-patterns -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: verifyDirectPattern
func @verifyDirectPattern() -> i32 {
  // CHECK-NEXT:  "test.legal_op_a"() {status = "Success"}
  %result = "test.illegal_op_a"() : () -> (i32)
  return %result : i32
}

// CHECK-LABEL: verifyLargerBenefit
func @verifyLargerBenefit() -> i32 {
  // CHECK-NEXT:  "test.legal_op_a"() {status = "Success"}
  %result = "test.illegal_op_c"() : () -> (i32)
  return %result : i32
}

// CHECK-LABEL: func @remap_input_1_to_0()
func @remap_input_1_to_0(i16)

// CHECK-LABEL: func @remap_input_1_to_1(%arg0: f64)
func @remap_input_1_to_1(%arg0: i64) {
  // CHECK-NEXT: "test.valid"{{.*}} : (f64)
  "test.invalid"(%arg0) : (i64) -> ()
}

// CHECK-LABEL: func @remap_input_1_to_N({{.*}}f16, {{.*}}f16)
func @remap_input_1_to_N(%arg0: f32) -> f32 {
 // CHECK-NEXT: "test.return"{{.*}} : (f16, f16) -> ()
 "test.return"(%arg0) : (f32) -> ()
}

// CHECK-LABEL: func @remap_input_1_to_N_remaining_use(%arg0: f16, %arg1: f16)
func @remap_input_1_to_N_remaining_use(%arg0: f32) {
  // CHECK-NEXT: [[CAST:%.*]] = "test.cast"(%arg0, %arg1) : (f16, f16) -> f32
  // CHECK-NEXT: "work"([[CAST]]) : (f32) -> ()
  "work"(%arg0) : (f32) -> ()
}

// CHECK-LABEL: func @remap_input_to_self
func @remap_input_to_self(%arg0: index) {
  // CHECK-NOT: test.cast
  // CHECK: "work"
  "work"(%arg0) : (index) -> ()
}

// CHECK-LABEL: func @remap_multi(%arg0: f64, %arg1: f64) -> (f64, f64)
func @remap_multi(%arg0: i64, %unused: i16, %arg1: i64) -> (i64, i64) {
 // CHECK-NEXT: "test.valid"{{.*}} : (f64, f64)
 "test.invalid"(%arg0, %arg1) : (i64, i64) -> ()
}

// CHECK-LABEL: func @no_remap_nested
func @no_remap_nested() {
  // CHECK-NEXT: "foo.region"
  "foo.region"() ({
    // CHECK-NEXT: ^bb0(%{{.*}}: i64, %{{.*}}: i16, %{{.*}}: i64):
    ^bb0(%i0: i64, %unused: i16, %i1: i64):
      // CHECK-NEXT: "test.valid"{{.*}} : (i64, i64)
      "test.invalid"(%i0, %i1) : (i64, i64) -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @remap_moved_region_args
func @remap_moved_region_args() {
  // CHECK-NEXT: return
  // CHECK-NEXT: ^bb1(%{{.*}}: f64, %{{.*}}: f64, %{{.*}}: f16, %{{.*}}: f16):
  // CHECK-NEXT: "test.cast"{{.*}} : (f16, f16) -> f32
  // CHECK-NEXT: "test.valid"{{.*}} : (f64, f64, f32)
  "test.region"() ({
    ^bb1(%i0: i64, %unused: i16, %i1: i64, %2: f32):
      "test.invalid"(%i0, %i1, %2) : (i64, i64, f32) -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @remap_drop_region
func @remap_drop_region() {
  // CHECK-NEXT: return
  // CHECK-NEXT: }
  "test.drop_op"() ({
    ^bb1(%i0: i64, %unused: i16, %i1: i64, %2: f32):
      "test.invalid"(%i0, %i1, %2) : (i64, i64, f32) -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @dropped_input_in_use
func @dropped_input_in_use(%arg: i16, %arg2: i64) {
  // CHECK-NEXT: "test.cast"{{.*}} : () -> i16
  // CHECK-NEXT: "work"{{.*}} : (i16)
  "work"(%arg) : (i16) -> ()
}

// CHECK-LABEL: func @up_to_date_replacement
func @up_to_date_replacement(%arg: i8) -> i8 {
  // CHECK-NEXT: return
  %repl_1 = "test.rewrite"(%arg) : (i8) -> i8
  %repl_2 = "test.rewrite"(%repl_1) : (i8) -> i8
  return %repl_2 : i8
}

// -----

func @fail_to_convert_illegal_op() -> i32 {
  // expected-error@+1 {{failed to legalize operation 'test.illegal_op_f'}}
  %result = "test.illegal_op_f"() : () -> (i32)
  return %result : i32
}

// -----

func @fail_to_convert_illegal_op_in_region() {
  // expected-error@+1 {{failed to legalize operation 'test.region_builder'}}
  "test.region_builder"() : () -> ()
  return
}

// -----

// Check that the entry block arguments of a region are untouched in the case
// of failure.

// CHECK-LABEL: func @fail_to_convert_region
func @fail_to_convert_region() {
  // CHECK-NEXT: "test.drop_op"
  // CHECK-NEXT: ^bb{{.*}}(%{{.*}}: i64):
  "test.drop_op"() ({
    ^bb1(%i0: i64):
      // expected-error@+1 {{failed to legalize operation 'test.region_builder'}}
      "test.region_builder"() : () -> ()
      "test.valid"() : () -> ()
  }) : () -> ()
  return
}

// -----

// Test parsing of an op with multiple region arguments, and without a
// delimiter.

// CHECK-LABEL: func @op_with_region_args
func @op_with_region_args() {
  // CHECK: "test.polyfor"() ( {
  // CHECK-NEXT: ^bb{{.*}}(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
  test.polyfor %i, %j, %k {
    "foo"() : () -> ()
  }
  return
}
