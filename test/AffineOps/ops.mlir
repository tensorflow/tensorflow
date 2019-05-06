// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s -mlir-print-op-generic | FileCheck -check-prefix=GENERIC %s

// Check that the attributes for the affine operations are round-tripped.
// Check that `affine.terminator` is visible in the generic form.
// CHECK-LABEL: @empty
func @empty() {
  // CHECK: affine.for %i
  // CHECK-NEXT: } {some_attr: true}
  //
  // GENERIC:      "affine.for"()
  // GENERIC-NEXT: ^bb1(%i0: index):
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: })
  affine.for %i = 0 to 10 {
  } {some_attr: true}

  // CHECK: affine.if
  // CHECK-NEXT: } {some_attr: true}
  //
  // GENERIC:      "affine.if"()
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: },  {
  // GENERIC-NEXT: })
  affine.if () : () () {
  } {some_attr: true}

  // CHECK: } else {
  // CHECK: } {some_attr: true}
  //
  // GENERIC:      "affine.if"()
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: },  {
  // GENERIC-NEXT:   "foo"() : () -> ()
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: })
  affine.if () : () () {
  } else {
    "foo"() : () -> ()
  } {some_attr: true}

  return
}

// Check that an explicit affine terminator is not printed in custom format.
// Check that no extra terminator is introduced.
// CHEKC-LABEL: @affine_terminator
func @affine_terminator() {
  // CHECK: affine.for %i
  // CHECK-NEXT: }
  //
  // GENERIC:      "affine.for"() ( {
  // GENERIC-NEXT: ^bb1(%i0: index):	// no predecessors
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: }) {lower_bound: #map0, step: 1 : index, upper_bound: #map1} : () -> ()
  affine.for %i = 0 to 10 {
    "affine.terminator"() : () -> ()
  }
  return
}
