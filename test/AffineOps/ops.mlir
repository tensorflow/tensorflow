// RUN: mlir-opt %s | FileCheck %s

// Check that the attributes for the affine operations are round-tripped.
func @attributes() {
  // CHECK: for %i
  // CHECK-NEXT: } {some_attr: true}
  for %i = 0 to 10 {
  } {some_attr: true}

  // CHECK: if
  // CHECK-NEXT: } {some_attr: true}
  if () : () () {
  } {some_attr: true}

  // CHECK: } else {
  // CHECK: } {some_attr: true}
  if () : () () {
  } else {
    "foo"() : () -> ()
  } {some_attr: true}

  return
}
