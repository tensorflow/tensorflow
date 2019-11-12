// RUN: mlir-opt -split-input-file %s | FileCheck %s
// RUN: mlir-opt %s -mlir-print-op-generic | FileCheck -check-prefix=GENERIC %s

// Check that the attributes for the affine operations are round-tripped.
// Check that `affine.terminator` is visible in the generic form.
// CHECK-LABEL: @empty
func @empty() {
  // CHECK: affine.for
  // CHECK-NEXT: } {some_attr = true}
  //
  // GENERIC:      "affine.for"()
  // GENERIC-NEXT: ^bb0(%{{.*}}: index):
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: })
  affine.for %i = 0 to 10 {
  } {some_attr = true}

  // CHECK: affine.if
  // CHECK-NEXT: } {some_attr = true}
  //
  // GENERIC:      "affine.if"()
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: },  {
  // GENERIC-NEXT: })
  affine.if () : () () {
  } {some_attr = true}

  // CHECK: } else {
  // CHECK: } {some_attr = true}
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
  } {some_attr = true}

  return
}

// Check that an explicit affine terminator is not printed in custom format.
// Check that no extra terminator is introduced.
// CHECK-LABEL: @affine_terminator
func @affine_terminator() {
  // CHECK: affine.for
  // CHECK-NEXT: }
  //
  // GENERIC:      "affine.for"() ( {
  // GENERIC-NEXT: ^bb0(%{{.*}}: index):	// no predecessors
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
  affine.for %i = 0 to 10 {
    "affine.terminator"() : () -> ()
  }
  return
}

// -----

// CHECK-DAG: #[[MAP0:map[0-9]+]] = (d0)[s0] -> (1000, d0 + 512, s0)
// CHECK-DAG: #[[MAP1:map[0-9]+]] = (d0, d1)[s0] -> (d0 - d1, s0 + 512)
// CHECK-DAG: #[[MAP2:map[0-9]+]] = ()[s0, s1] -> (s0 - s1, 11)
// CHECK-DAG: #[[MAP3:map[0-9]+]] = () -> (77, 78, 79)

// CHECK-LABEL: @affine_min
func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: affine.min #[[MAP0]](%arg0)[%arg1]
  %0 = affine.min (d0)[s0] -> (1000, d0 + 512, s0) (%arg0)[%arg1]
  // CHECK: affine.min #[[MAP1]](%arg0, %arg1)[%arg2]
  %1 = affine.min (d0, d1)[s0] -> (d0 - d1, s0 + 512) (%arg0, %arg1)[%arg2]
  // CHECK: affine.min #[[MAP2]]()[%arg1, %arg2]
  %2 = affine.min ()[s0, s1] -> (s0 - s1, 11) ()[%arg1, %arg2]
  // CHECK: affine.min #[[MAP3]]()
  %3 = affine.min ()[] -> (77, 78, 79) ()[]
  return
}
