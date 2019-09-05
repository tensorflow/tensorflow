// RUN: mlir-opt %s -test-inline | FileCheck %s

// CHECK-LABEL: func @inline_with_arg
func @inline_with_arg(%arg0 : i32) -> i32 {
  // CHECK-NEXT: %[[ADD:.*]] = addi %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT: return %[[ADD]] : i32
  %fn = "test.functional_region_op"() ({
  ^bb0(%a : i32):
    %b = addi %a, %a : i32
    "test.return"(%b) : (i32) -> ()
  }) : () -> ((i32) -> i32)

  %0 = call_indirect %fn(%arg0) : (i32) -> i32
  return %0 : i32
}

// CHECK-LABEL: func @no_inline_invalid_nested_operation
func @no_inline_invalid_nested_operation() {
  // CHECK: call_indirect

  // test.region is analyzed recursively, so it must not have an invalid op.

  %fn = "test.functional_region_op"() ({
    "test.region"() ({
      "foo.noinline_operation"() : () -> ()
    }) : () -> ()
    "test.return"() : () -> ()
  }) : () -> (() -> ())

  call_indirect %fn() : () -> ()
  return
}

// CHECK-LABEL: func @inline_ignore_invalid_nested_operation
func @inline_ignore_invalid_nested_operation() {
  // CHECK-NOT: call_indirect

  // test.functional_region_op is not analyzed recursively, so it may have an
  // invalid op.

  %fn = "test.functional_region_op"() ({
    %internal_fn = "test.functional_region_op"() ({
      "foo.noinline_operation"() : () -> ()
    }) : () -> (() -> ())
    "test.return"() : () -> ()
  }) : () -> (() -> ())

  call_indirect %fn() : () -> ()
  return
}

// CHECK-LABEL: func @no_inline_invalid_dest_region
func @no_inline_invalid_dest_region() {
  // CHECK: call_indirect

  // foo.unknown_region is unknown, so we can't inline into it.

  "foo.unknown_region"() ({
    %fn = "test.functional_region_op"() ({
      "test.return"() : () -> ()
    }) : () -> (() -> ())
    call_indirect %fn() : () -> ()
    "test.return"() : () -> ()
  }) : () -> ()

  return
}
