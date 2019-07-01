// RUN: mlir-opt -test-legalize-patterns %s | FileCheck %s

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

// CHECK-LABEL: func @remap_input_1_to_N(%arg0: f16, %arg1: f16)
func @remap_input_1_to_N(%arg0: f32) -> f32 {
 // TODO: this is temporarily disabled because the rewriter does not
 // change "test.invalid" into "test.valid" that takes two operands,
 // making the use of the original operand persist and materializing
 // the type conversion.
 // X-CHECK-NEXT: "test.valid"(%arg0, %arg1) : (f16, f16) -> ()
 "test.invalid"(%arg0) : (f32) -> ()
}

// CHECK-LABEL: func @remap_input_1_to_N_remaining_use(%arg0: f16, %arg1: f16)
func @remap_input_1_to_N_remaining_use(%arg0: f32) {
  // CHECK-NEXT: [[CAST:%.*]] = "test.cast"(%arg0, %arg1) : (f16, f16) -> f32
  // CHECK-NEXT: "work"([[CAST]]) : (f32) -> ()
  "work"(%arg0) : (f32) -> ()
}

// CHECK-LABEL: func @remap_multi(%arg0: f64, %arg1: f64) -> (f64, f64)
func @remap_multi(%arg0: i64, %unused: i16, %arg1: i64) -> (i64, i64) {
 // CHECK-NEXT: "test.valid"{{.*}} : (f64, f64)
 "test.invalid"(%arg0, %arg1) : (i64, i64) -> ()
}

// CHECK-LABEL: func @remap_nested
func @remap_nested() {
  // CHECK-NEXT: "foo.region"
  "foo.region"() ({
    // CHECK-NEXT: ^bb1(%i0: f64, %i1: f64):
    ^bb1(%i0: i64, %unused: i16, %i1: i64):
      // CHECK-NEXT: "test.valid"{{.*}} : (f64, f64)
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
